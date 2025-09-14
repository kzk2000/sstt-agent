import os
from typing import List
from typing import Literal
import re
from typing import Set

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, confloat


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
        "Return only a minimal closed graph that captures the essential causal and semantic structure, "
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

    Answering rules:
    - Follow causal chains (LEADS-TO, EXPRESSES) to explain outcomes.
    - Give more weight to higher-intent nodes when deciding what matters.
    - Use low-intent nodes only if they clarify a causal chain, not as main points.
    - Provide a clear and concise answer to the user’s question in 1–2 sentences.
    - Do not paraphrase the context; instead, reason explicitly from the graph structure.
    """)


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


# --- Functions to manage memory ---
def write_memory(doc: str, g: Graph):
    """
    Distill a graph into one or more MemoryEntry objects.
    - Situation = source node label
    - Suggestion = relation + target node label(s)
    """
    for edge in g.edges:
        src_label = next((n.label for n in g.nodes if n.id == edge.source), edge.source)
        tgt_label = next((n.label for n in g.nodes if n.id == edge.target), edge.target)

        situation = src_label
        suggestion = f"{edge.relation} → {tgt_label}"

        entry = MemoryEntry(situation=situation, suggestion=suggestion, graph=g)
        memory_store.append(entry)


def read_memory(query: str) -> List[MemoryEntry]:
    """Retrieve relevant abstract concepts (simplified for demo)."""
    return [m for m in memory_store if any(w in query for w in m.situation.split())]


# --- Example run ---
doc = """
When retail flow is heavy, toxic takers reduce activity,
which causes market makers to narrow spreads.
Narrow spreads express confidence in liquidity.
"""

result = graph_agent.run_sync(doc)
g: Graph = result.output
print(g.model_dump_json(indent=2))

# --- Agent with memory ---
memory_store: List[MemoryEntry] = []
# write to memory
write_memory(doc, g)

# read memory for a new query
print(read_memory("What happens when retail flow is heavy?"))

if False:
    question = "What happens to spreads when retail trading stops?"
    query = graph_to_reasoning_prompt(g) + f"\n\n### User question:\n{question}"
    print(query)
    answer = question_agent.run_sync(query)
    print(answer.output)
