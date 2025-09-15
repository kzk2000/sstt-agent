"""
Pydantic AI code_agent that:
- Takes a reasoning plan (e.g., "load → filter → aggregate → rank")
- Uses tools to lazily look up relevant FunctionCards (to avoid huge prompts)
- Chooses concrete functions per step and generates runnable Python code
- Maintains a DataState with grain/columns to enforce basic schema constraints

You can swap the model (e.g., 'google-gla:gemini-1.5-flash' or 'gpt-4o') without changing the tools.

Usage (pseudo):
    deps = Deps(registry=FUNCTION_CARDS)
    result = code_agent.run_sync(
        deps,
        {
            "user_question": "Top 5 clients by margin last year",
            "plan": "load → filter → aggregate → rank"
        }
    )
    print(result.data)  # full Python code as a string
"""
from __future__ import annotations

import os
import re
from typing import Dict, Optional
from typing import List
from typing import Literal
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


# ------------------------------
# Function metadata models
# ------------------------------
class ColumnSpec(BaseModel):
    name: str
    dtype: str
    role: str               # 'key' | 'metric' | 'dimension' | 'flag'
    units: Optional[str] = None
    meaning: str

class ReturnSpec(BaseModel):
    grain: str              # e.g. 'row per client'
    columns: List[ColumnSpec]

class FunctionCard(BaseModel):
    name: str
    step: Literal['load','filter','transform','aggregate','rank','export']
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
    # runtime state
    state_grain: str = 'unknown'
    state_columns: Dict[str, ColumnSpec] = {}

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
    #model_name="gemini-2.0-flash-lite",
    provider=OpenAIProvider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ['GOOGLE_API_KEY'],
    ),
)

code_agent = Agent[Deps,str](
    model=model,
    system_prompt=(
        "You are a Python code generation agent for a small analytics package.\n"
        "Inputs: a user question and an abstract reasoning plan (e.g., 'load → filter → aggregate → rank').\n"
        "Tools let you look up functions on demand.\n\n"
        "Your job:\n"
        "1) For each plan step, call list_functions_for_step to see available functions.\n"
        "2) Pick the most relevant candidate(s) by calling get_function_card to read doc, signature, tags, and schema.\n"
        "3) Ensure requires_columns are present in the current state before selecting a function; if not, prefer an alternative that fits.\n"
        "4) After choosing a function, call apply_function to update DataState (grain+columns).\n"
        "5) Produce COMPLETE, runnable Python: imports + function calls following the plan order.\n\n"
        "Rules:\n"
        "- Use ONLY the functions the tools return. Do NOT invent functions.\n"
        "- Keep parameters consistent with signatures.\n"
        "- If multiple candidates are viable, choose the one whose tags/return_spec match the user's goal best.\n"
        "- Return ONLY a Python code block as final answer."
    ),
)

# ------------------------------
# Tools
# ------------------------------
@code_agent.tool(retries=5)
def list_functions_for_step(ctx: RunContext[Deps], step: str) -> List[str]:
    """List function names available for a given plan step."""
    return [fc.name for fc in ctx.deps.step_index().get(step, [])]

@code_agent.tool(retries=5)
def get_function_card(ctx: RunContext[Deps], name: str, fields: Optional[List[str]] = None) -> dict:
    """Get a (possibly trimmed) FunctionCard by name. Use `fields` to keep responses small."""
    fc = next((f for f in ctx.deps.registry if f.name == name), None)
    if not fc:
        return {}
    data = fc.model_dump()
    if fields:
        # allow nested field paths like 'return_spec.grain' or 'return_spec.columns'
        trimmed: dict = {}
        for f in fields:
            cur = data
            dest = trimmed
            parts = f.split('.')
            for i, p in enumerate(parts):
                if p not in cur:
                    cur = None
                    break
                if i == len(parts) - 1:
                    dest[p] = cur[p]
                else:
                    dest = dest.setdefault(p, {})
                    cur = cur[p]
        return trimmed
    return data

@code_agent.tool(retries=5)
def compatible_with_state(ctx: RunContext[Deps], requires_columns: List[str]) -> List[str]:
    """Return missing required columns for the current state (empty list means compatible)."""
    missing = [c for c in requires_columns if c not in ctx.deps.state_columns]
    return missing

@code_agent.tool(retries=5)
def apply_function(ctx: RunContext[Deps], name: str) -> dict:
    """Update DataState (grain & columns) using the function's return_spec. Return the new state summary."""
    fc = next((f for f in ctx.deps.registry if f.name == name), None)
    if not fc:
        return {"error": f"unknown function {name}"}
    ctx.deps.state_grain = fc.return_spec.grain
    ctx.deps.state_columns = {c.name: c for c in fc.return_spec.columns}
    return {
        "grain": ctx.deps.state_grain,
        "columns": list(ctx.deps.state_columns.keys()),
    }

# (Optional) tiny helper to propose a call template based on signature
@code_agent.tool(retries=2)
def call_template(ctx: RunContext[Deps], name: str) -> str:
    """Return a minimal call template string based on the function signature (best effort)."""
    fc = next((f for f in ctx.deps.registry if f.name == name), None)
    if not fc:
        return ''
    sig = fc.signature
    # naive: extract name and args inside parentheses
    m = re.match(r"(\w+)\((.*)\)", sig)
    if not m:
        return f"{name}()"
    fname, args = m.group(1), m.group(2)
    # strip annotations and defaults into simple arg names
    parts = []
    for a in args.split(','):
        a = a.strip()
        if not a:
            continue
        # drop type annotations and defaults
        a = a.split(':')[0].split('=')[0].strip()
        if a == 'df':
            parts.append('df')
        elif a.lower() in ('start_date','end_date'):
            parts.append("'2024-01-01'" if a.lower()=='start_date' else "'2024-12-31'")
        elif a.lower() in ('metric','by','n','min_revenue'):
            # placeholders the agent can edit
            defaults = {
                'metric': "'revenue'",
                'by': "'client_id'",
                'n': '5',
                'min_revenue': '0.0',
            }
            parts.append(defaults.get(a.lower(), ''))
        else:
            parts.append('')
    return f"{fname}({', '.join([p for p in parts if p!=''])})"

# ------------------------------
# Demo registry (same toy functions as earlier mock)
# ------------------------------
FUNCTION_CARDS: List[FunctionCard] = [
    FunctionCard(
        name='load_data',
        step='load',
        signature='load_data(start_date: str, end_date: str) -> DataFrame',
        doc='Load client-level transactions between start_date and end_date.',
        requires_columns=[],
        guarantees_columns=['client_id','segment','revenue','margin'],
        return_spec=ReturnSpec(
            grain='row per transaction',
            columns=[
                ColumnSpec(name='client_id', dtype='str', role='key', units=None, meaning='unique client identifier'),
                ColumnSpec(name='segment', dtype='str', role='dimension', units=None, meaning='client segment name'),
                ColumnSpec(name='revenue', dtype='float', role='metric', units='USD', meaning='gross revenue amount'),
                ColumnSpec(name='margin', dtype='float', role='metric', units='USD', meaning='gross margin amount'),
            ]
        ),
        examples=["df = load_data('2024-01-01','2024-12-31')"],
        tags=['load','transactions','clients']
    ),
    FunctionCard(
        name='filter_clients',
        step='filter',
        signature='filter_clients(df: DataFrame, min_revenue: float | None = None) -> DataFrame',
        doc='Filter to active/eligible clients; optional min revenue threshold.',
        requires_columns=['client_id','revenue'],
        guarantees_columns=['client_id','segment','revenue','margin'],
        return_spec=ReturnSpec(
            grain='row per transaction',
            columns=[
                ColumnSpec(name='client_id', dtype='str', role='key', units=None, meaning='unique client identifier'),
                ColumnSpec(name='segment', dtype='str', role='dimension', units=None, meaning='client segment name'),
                ColumnSpec(name='revenue', dtype='float', role='metric', units='USD', meaning='gross revenue amount'),
                ColumnSpec(name='margin', dtype='float', role='metric', units='USD', meaning='gross margin amount'),
            ]
        ),
        examples=['df = filter_clients(df, min_revenue=100.0)'],
        tags=['filter','eligibility']
    ),
    FunctionCard(
        name='aggregate_by_metric',
        step='aggregate',
        signature='aggregate_by_metric(df: DataFrame, metric: str, by: str = "client_id") -> DataFrame',
        doc='Aggregate a metric at a chosen grain (default per client).',
        requires_columns=['client_id'],
        guarantees_columns=['client_id','metric_value'],
        return_spec=ReturnSpec(
            grain='row per client',
            columns=[
                ColumnSpec(name='client_id', dtype='str', role='key', units=None, meaning='unique client identifier'),
                ColumnSpec(name='metric_value', dtype='float', role='metric', units='USD', meaning='aggregated metric value for client'),
            ]
        ),
        examples=["df = aggregate_by_metric(df, metric='margin', by='client_id')"],
        tags=['aggregate','client','revenue','margin']
    ),
    FunctionCard(
        name='retrieve_top_clients',
        step='rank',
        signature='retrieve_top_clients(df: DataFrame, n: int = 5, by: str | None = "metric_value") -> DataFrame',
        doc='Return the top-n clients by a numeric column (default metric_value).',
        requires_columns=['client_id','metric_value'],
        guarantees_columns=['client_id','metric_value','rank'],
        return_spec=ReturnSpec(
            grain='row per client (top-n)',
            columns=[
                ColumnSpec(name='client_id', dtype='str', role='key', units=None, meaning='unique client identifier'),
                ColumnSpec(name='metric_value', dtype='float', role='metric', units='USD', meaning='metric used for ranking'),
                ColumnSpec(name='rank', dtype='int', role='dimension', units=None, meaning='1 is best'),
            ]
        ),
        examples=['top5 = retrieve_top_clients(df, n=5, by="metric_value")'],
        tags=['rank','top-n']
    ),
]

# ------------------------------
# Example runner (pseudo)
# ------------------------------
if __name__ == '__main__':
    deps = Deps(registry=FUNCTION_CARDS)
    @dataclass()
    class Ctx():
        deps = deps

    ctx = Ctx()
    question = "Top 5 clients by margin in 2025"
    #question = "Top 5 clients by revenue in Q2 2025"
    plan = "load → filter → aggregate → rank"
    # NOTE: Requires a configured pydantic-ai model provider

    prompt= f"""
### Reasoning plan:
{plan}

### User question:
{question}
"""
    result = code_agent.run_sync(prompt, deps=deps)
    print(result.output)

    result.all_messages()
    if False:
        import json
        print(json.dumps(json.loads(result.all_messages_json()), indent=2))