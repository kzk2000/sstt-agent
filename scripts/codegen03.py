"""
Pydantic AI code_agent with integrated search_functions(step, query):
- For each step in a reasoning plan, the agent calls search_functions(step, query).
- Then it inspects candidate functions using get_function_card, which by default returns only {signature, return_spec}.
- The agent uses this to decide how to call the function and what it returns.
- Maintains a DataState with grain/columns.
- apply_function logs finite-state transitions (state_before, function, state_after) to a trace.
- New tool get_trace exposes the trace so the agent can reason about its own history.
- Outputs runnable Python code.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


# ------------------------------
# Function metadata models
# ------------------------------
class ColumnSpec(BaseModel):
    name: str
    dtype: str
    role: str
    units: Optional[str] = None
    meaning: str


class ReturnSpec(BaseModel):
    grain: str
    columns: List[ColumnSpec]


class FunctionCard(BaseModel):
    name: str
    step: Literal['load', 'filter', 'transform', 'aggregate', 'rank', 'export']
    signature: str
    doc: str
    requires_columns: List[str]
    guarantees_columns: List[str]
    return_spec: ReturnSpec
    examples: List[str] = []
    tags: List[str] = []


# ------------------------------
# Registry + simple indexing
# ------------------------------
class Deps(BaseModel):
    registry: List[FunctionCard]
    state_grain: str = 'unknown'
    state_columns: Dict[str, ColumnSpec] = {}
    trace: List[dict] = []  # FSA-like trace of transitions

    def step_index(self) -> Dict[str, List[FunctionCard]]:
        idx: Dict[str, List[FunctionCard]] = {}
        for fc in self.registry:
            idx.setdefault(fc.step, []).append(fc)
        return idx


# ------------------------------
# Helper utils
# ------------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> set[str]:
    return set(w.lower() for w in TOKEN_RE.findall(text or ""))


# ------------------------------
# The Agent
# ------------------------------
model = OpenAIChatModel(
    model_name="gemini-2.5-flash",
    # model_name="gemini-2.0-flash-lite",
    provider=OpenAIProvider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ['GOOGLE_API_KEY'],
    ),
)

code_agent = Agent[Deps, str](
    model=model,
    system_prompt=(
        "You are a Python code generation agent for a small analytics package.\n"
        "Inputs: a user question and an abstract reasoning plan (e.g., 'load → filter → aggregate → rank').\n"
        "For each plan step, call search_functions(step, query) with the step and the user question.\n"
        "Then call get_function_card for the top candidates.\n\n"
        "Important: get_function_card returns only {signature, return_spec}.\n"
        "- Use signature to construct the call.\n"
        "- Use return_spec (grain + columns) to know what data comes out, and to ensure chaining works.\n\n"
        "Update DataState with apply_function.\n"
        "apply_function also logs each transition (state_before, function, state_after) to a trace.\n"
        "If you are uncertain about current state, you may call get_trace to inspect history.\n"
        "Finally, produce COMPLETE runnable Python (imports + calls).\n\n"
        "Rules:\n"
        "- Use ONLY functions returned by tools. Do NOT invent functions.\n"
        "- Keep parameters consistent with signatures.\n"
        "- Return ONLY a Python code block as final answer."
    )
)


# ------------------------------
# Tools
# ------------------------------
@code_agent.tool(retries=5)
def search_functions(ctx: RunContext[Deps], step: str, query: str, top_k: int = 5) -> List[dict]:
    """Semantic search restricted to a given plan step. Returns top_k {name, score}."""
    qtok = tokenize(query)
    scored = []
    for fc in ctx.deps.registry:
        if step and fc.step != step:
            continue
        text = f"{fc.name} {fc.doc} {' '.join(fc.tags)} " + \
               ' '.join(c.name for c in fc.return_spec.columns)
        ftok = tokenize(text)
        overlap = len(qtok & ftok) / max(1, len(qtok | ftok))
        scored.append((overlap, fc.name))
    scored.sort(reverse=True)
    return [{"name": name, "score": round(score, 3)} for score, name in scored[:top_k]]


@code_agent.tool(retries=5)
def get_function_card(ctx: RunContext[Deps], name: str) -> dict:
    """Return only signature + return_spec for the given function."""
    fc = next((f for f in ctx.deps.registry if f.name == name), None)
    if not fc:
        return {}
    retval = {
        "signature": fc.signature,
        "return_spec": fc.return_spec.model_dump()
    }
    print(50 * "*")
    print(retval)
    return retval


@code_agent.tool(retries=5)
def compatible_with_state(ctx: RunContext[Deps], requires_columns: List[str]) -> List[str]:
    return [c for c in requires_columns if c not in ctx.deps.state_columns]


@code_agent.tool(retries=5)
def apply_function(ctx: RunContext[Deps], name: str) -> dict:
    """Update DataState with the function's return_spec and log the transition (FSA trace)."""
    fc = next((f for f in ctx.deps.registry if f.name == name), None)
    if not fc:
        return {"error": f"unknown function {name}"}

    state_before = {"grain": ctx.deps.state_grain, "columns": list(ctx.deps.state_columns.keys())}
    ctx.deps.state_grain = fc.return_spec.grain
    ctx.deps.state_columns = {c.name: c for c in fc.return_spec.columns}
    state_after = {"grain": ctx.deps.state_grain, "columns": list(ctx.deps.state_columns.keys())}
    ctx.deps.trace.append({
        "function": name,
        "state_before": state_before,
        "state_after": state_after
    })
    return state_after


@code_agent.tool(retries=5)
def get_trace(ctx: RunContext[Deps]) -> List[dict]:
    """Return the current transition trace."""
    return ctx.deps.trace


# ------------------------------
# Demo registry (toy functions)
# ------------------------------
FUNCTION_CARDS: List[FunctionCard] = [
    FunctionCard(
        name='load_data', step='load', signature='load_data(start_date: str, end_date: str) -> DataFrame',
        doc='Load client-level transactions between start_date and end_date.',
        requires_columns=[], guarantees_columns=['client_id', 'segment', 'revenue', 'margin'],
        return_spec=ReturnSpec(grain='row per transaction', columns=[
            ColumnSpec(name='client_id', dtype='str', role='key', meaning='unique client id'),
            ColumnSpec(name='segment', dtype='str', role='dimension', meaning='client segment'),
            ColumnSpec(name='revenue', dtype='float', role='metric', units='USD', meaning='revenue amount'),
            ColumnSpec(name='margin', dtype='float', role='metric', units='USD', meaning='margin amount'),
        ])
    ),
    FunctionCard(
        name='filter_clients', step='filter',
        signature='filter_clients(df: DataFrame, min_revenue: float | None = None) -> DataFrame',
        doc='Filter to active clients.',
        requires_columns=['client_id', 'revenue'], guarantees_columns=['client_id', 'segment', 'revenue', 'margin'],
        return_spec=ReturnSpec(grain='row per transaction', columns=[
            ColumnSpec(name='client_id', dtype='str', role='key', meaning='unique client id'),
            ColumnSpec(name='segment', dtype='str', role='dimension', meaning='client segment'),
            ColumnSpec(name='revenue', dtype='float', role='metric', units='USD', meaning='revenue amount'),
            ColumnSpec(name='margin', dtype='float', role='metric', units='USD', meaning='margin amount'),
        ])
    ),
    FunctionCard(
        name='aggregate_by_metric', step='aggregate',
        signature='aggregate_by_metric(df: DataFrame, metric: str, by: str = "client_id") -> DataFrame',
        doc='Aggregate a metric per client.',
        requires_columns=['client_id'], guarantees_columns=['client_id', 'metric_value'],
        return_spec=ReturnSpec(grain='row per client', columns=[
            ColumnSpec(name='client_id', dtype='str', role='key', meaning='unique client id'),
            ColumnSpec(name='metric_value', dtype='float', role='metric', units='USD', meaning='aggregated metric'),
        ])
    ),
    FunctionCard(
        name='retrieve_top_clients', step='rank',
        signature='retrieve_top_clients(df: DataFrame, n: int = 5, by: str | None = "metric_value") -> DataFrame',
        doc='Return the top-n clients.',
        requires_columns=['client_id', 'metric_value'], guarantees_columns=['client_id', 'metric_value', 'rank'],
        return_spec=ReturnSpec(grain='row per client (top-n)', columns=[
            ColumnSpec(name='client_id', dtype='str', role='key', meaning='unique client id'),
            ColumnSpec(name='metric_value', dtype='float', role='metric', units='USD',
                       meaning='metric used for ranking'),
            ColumnSpec(name='rank', dtype='int', role='dimension', meaning='1 is best'),
        ])
    ),
]

# ------------------------------
# Example runner
# ------------------------------
if __name__ == '__main__':
    deps = Deps(registry=FUNCTION_CARDS)
    question = "Top 5 clients by margin last year"
    plan = "load → filter → aggregate → rank"

    prompt = f"""
    ### Reasoning plan:
    {plan}

    ### User question:
    {question}
    """
    result = code_agent.run_sync(prompt, deps=deps)
    print(result.output)
    print("\n--- Transition Trace ---")
    for t in deps.trace:
        print(t)

    if False:
        result.all_messages()
