import os
import re
from typing import List, Literal, Set, Tuple

from pydantic import BaseModel, confloat
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


class Node(BaseModel):
    id: str
    type: Literal["event", "thing", "concept"]  # only valid γ(3,4) node types
    label: str
    intentionality: confloat(ge=0.0, le=1.0)  # must be between 0.0 and 1.0


class Edge(BaseModel):
    source: str
    target: str
    relation: Literal["NEAR", "LEADS-TO", "CONTAINS", "EXPRESSES"]  # only valid γ(3,4) edge types


class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


# --- Memory entry inspired by ArcMemo/Memento ---
class MemoryEntry(BaseModel):
    situation: str  # abstract condition (ArcMemo: situation X)
    suggestion: str  # general rule/concept (ArcMemo: do Y)
    graph: Graph | None  # optional γ(3,4) subgraph that encodes it


# ----------------------
# Agent definition
# ----------------------
model = OpenAIChatModel(
    model_name="gemini-2.5-pro",
    provider=OpenAIProvider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ['GOOGLE_API_KEY'],
    ),
)

graph_agent = Agent[None, Graph](
    model=model,
    output_type=Graph,
    system_prompt=(
        "You are a knowledge graph extractor using the γ(3,4) Semantic Spacetime model.\n\n"
        "Parse text into nodes and edges.\n\n"
        "Node types (3):\n"
        "- Event: something that happens in time (actions, processes, state changes).\n"
        "  Example: 'toxic takers reduce activity', 'market makers narrow spreads'.\n"
        "- Thing: a persistent object or entity (actors, instruments, resources).\n"
        "  Example: 'market makers', 'liquidity', 'toxic takers'.\n"
        "- Concept: an abstract property, quality, or idea.\n"
        "  Example: 'confidence in liquidity', 'risk aversion'.\n\n"
        "Edge types (4):\n"
        "- NEAR: contextual or co-occurrence relation.\n"
        "- LEADS-TO: causal or temporal succession relation.\n"
        "- CONTAINS: part-of, membership, or scope relation.\n"
        "- EXPRESSES: representational, symbolic, or signal relation.\n\n"
        "Each node must include an `intentionality` score between 0.0 and 1.0.\n"
        "Interpret intentionality as an entropy or information score:\n"
        "- High entropy (0.7–1.0): rare, surprising, or information-rich phrases — core ideas.\n"
        "- Medium entropy (0.4–0.6): moderately informative, partly contextual.\n"
        "- Low entropy (0.0–0.3): predictable, generic background with little information.\n\n"
        "Use the following heuristics when estimating intentionality:\n"
        "- Increase score for:\n"
        "  * Rare or unusual words/phrases (higher information content / work cost).\n"
        "  * Longer or multi-word expressions that are not generic.\n"
        "  * Concepts that recur irregularly across the text (burstiness).\n"
        "- Decrease score for:\n"
        "  * Very common or generic words.\n"
        "  * Phrases repeated too frequently in a short span (low coherence).\n\n"
        "Return only a minimal closed graph that captures the essential causal and semantic structure,\n"
        "not raw n-grams. Do not include unnecessary nodes.\n\n"
    ),
)

question_agent = Agent(
    model=model,
    system_prompt="""
    You are a helpful Q&A agent.
    You are given a reasoning graph built using the γ(3,4) Semantic Spacetime model.

    Interpretation rules:
    - Nodes represent events, things, or concepts.
    - Edges show relationships: NEAR, LEADS-TO, CONTAINS, or EXPRESSES.
    - Each node has an intentionality score in [0.0–1.0], which reflects its importance:
      * High (0.7–1.0): core, information-rich ideas that should anchor your reasoning.
      * Medium (0.4–0.6): moderately relevant, use when needed for clarity.
      * Low (0.0–0.3): background context, usually ignore unless directly tied into a causal chain.

    Reasoning rules:
    - For normal questions: follow causal chains (LEADS-TO, EXPRESSES) to explain outcomes.
    - For "what if" or counterfactual questions:
      * Identify the node that matches the changed condition.
      * Remove or negate that node from the causal chain.
      * Infer how the downstream nodes (effects) would change.
    - Give more weight to higher-intent nodes when deciding what matters.
    - Use low-intent nodes only if they clarify a causal chain, not as main points.

    Reasoning template (always follow internally before answering):
    1. Identify the relevant starting condition(s) from the graph.
    2. Trace through the causal/expressive edges.
    3. Note what happens downstream if the condition holds or is removed.
    4. Summarize the outcome.

    Answering rules:
    - Provide a clear and concise final answer in 1–2 sentences.
    - Express the outcome directly, without restating all nodes or edges.
    - Do not paraphrase the graph mechanically; explain the implication of the causal chain.
    """
)


def slugify(label: str) -> str:
    """Canonicalize labels into safe lowercase IDs."""
    return re.sub(r'[^a-z0-9_]', '', label.lower().replace(" ", "_"))


def clean_graph(g: Graph) -> Graph:
    """Post-process a Graph object:
    - Slugify node IDs
    - Deduplicate nodes and edges
    - Keep only minimal unique structure
    """
    seen_nodes: Set[str] = set()
    cleaned_nodes: list[Node] = []
    id_map = {}

    # --- Canonicalize nodes ---
    for n in g.nodes:
        slug = slugify(n.label)
        if slug not in seen_nodes:
            seen_nodes.add(slug)
            cleaned_nodes.append(Node(id=slug, type=n.type, label=n.label))
        id_map[n.id] = slug  # remap original ID to slug

    # --- Canonicalize + deduplicate edges ---
    seen_edges: Set[tuple] = set()
    cleaned_edges: list[Edge] = []
    for e in g.edges:
        src = id_map.get(e.source, slugify(e.source))
        tgt = id_map.get(e.target, slugify(e.target))
        edge_key = (src, tgt, e.relation)
        if edge_key not in seen_edges and src != tgt:
            seen_edges.add(edge_key)
            cleaned_edges.append(Edge(source=src, target=tgt, relation=e.relation))

    return Graph(nodes=cleaned_nodes, edges=cleaned_edges)


def graph_to_reasoning_prompt(g: Graph) -> str:
    id2label = {n.id: n.label for n in g.nodes}
    id2type = {n.id: n.type for n in g.nodes}

    lines = ["### Context: Retrieved Subgraph", "Nodes:"]
    for n in g.nodes:
        lines.append(f"- {n.label} ({n.type}, intentionality={n.intentionality:.2f})")

    lines.append("\nEdges:")
    for e in g.edges:
        src = id2label.get(e.source, e.source)
        tgt = id2label.get(e.target, e.target)
        lines.append(f"- {src} -[{e.relation}]-> {tgt}")

    lines += [
        "",
        "### Instruction",
        "Answer the user’s question by reasoning over the nodes and edges in this graph.",
        "Pay more attention to nodes with higher intentionality (closer to 1.0),",
        "and treat low-intent nodes (<0.3) as background context.",
        "Follow causal chains instead of paraphrasing."
    ]
    return "\n".join(lines)


class AxiomViolation(Exception):
    """Raised when a graph violates a hard axiom."""


# --- Axioms ---

def axiom_no_cycles_in_leads_to(g: Graph) -> List[Tuple[str, str]]:
    """A1: LEADS-TO edges must form an acyclic structure."""
    violations = []
    visited, stack = set(), set()

    def dfs(node_id: str):
        if node_id in stack:
            return True
        stack.add(node_id)
        for e in g.edges:
            if e.source == node_id and e.relation == "LEADS-TO":
                if dfs(e.target):
                    violations.append((e.source, e.target))
        stack.remove(node_id)
        visited.add(node_id)
        return False

    for n in g.nodes:
        if n.id not in visited:
            dfs(n.id)
    return violations


def axiom_no_near_as_only_link(g: Graph) -> List[Tuple[str, str]]:
    """A2: NEAR edges cannot be the sole connection between two events."""
    violations = []
    for e in g.edges:
        if e.relation == "NEAR":
            src_type = next((n.type for n in g.nodes if n.id == e.source), None)
            tgt_type = next((n.type for n in g.nodes if n.id == e.target), None)
            if src_type == "event" and tgt_type == "event":
                # Check if there’s *also* a causal/expressive link
                has_causal = any(
                    ee.source == e.source and ee.target == e.target and ee.relation in {"LEADS-TO", "EXPRESSES"}
                    for ee in g.edges
                )
                if not has_causal:
                    violations.append((e.source, e.target))
    return violations


def axiom_no_self_loops(g: Graph) -> List[Tuple[str, str, str]]:
    """A3: No node should point to itself."""
    return [(e.source, e.target, e.relation) for e in g.edges if e.source == e.target]


# --- Reinforcement mechanism (Memento-style) ---

def reinforce_edge(g: Graph, path: List[str], increment: float = 0.05):
    """
    Strengthen edges in a path by increasing intentionality of source & target nodes.
    - increment: how much to boost per traversal (cap at 1.0)
    """
    for nid in path:
        node = next((n for n in g.nodes if n.id == nid), None)
        if node:
            new_val = min(1.0, node.intentionality + increment)
            node.intentionality = new_val


def validate_graph(g: Graph) -> dict:
    """Run all axioms and return a report."""
    report = {
        "no_cycles_in_leads_to": axiom_no_cycles_in_leads_to(g),
        "no_near_as_only_link": axiom_no_near_as_only_link(g),
        "no_self_loops": axiom_no_self_loops(g),
    }
    return {k: v for k, v in report.items() if v}  # returns violations


# --- Example run ---
doc = """
When retail flow is heavy, toxic takers reduce activity,
which causes market makers to narrow spreads.
Narrow spreads express confidence in liquidity.
"""

result = graph_agent.run_sync(doc)
g: Graph = result.output
# print(g.model_dump_json(indent=2))
print(graph_to_reasoning_prompt(g))

violations = validate_graph(g)
if violations:
    print("Axiom violations:", violations)
else:
    print("Graph is valid.")

if False:
    question = "What happens to spreads when retail trading stops?"
    #    question = "What happens when there are very few toxic takers, and what impact might this have on retail trading?"

    query = graph_to_reasoning_prompt(g) + f"\n\n### User question:\n{question}"
    print(query)
    answer = question_agent.run_sync(query)
    print(50 * "*")
    print(answer.output)
