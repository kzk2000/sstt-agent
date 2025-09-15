"""
End-to-end mock of an SSTorytime-style reasoning + grounding stack
for a small Python package. No external LLM calls — fully deterministic.

This version fixes schema alignment so the demo runs cleanly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Tuple
import re

# ------------------------------
# 1) Function metadata layer
# ------------------------------
@dataclass
class ColumnSpec:
    name: str
    dtype: str
    role: str
    units: str | None
    meaning: str

@dataclass
class ReturnSpec:
    grain: str
    columns: List[ColumnSpec]

@dataclass
class FunctionCard:
    name: str
    step: Literal['load','filter','transform','aggregate','rank','export']
    signature: str
    doc: str
    requires_columns: List[str]
    guarantees_columns: List[str]
    return_spec: ReturnSpec
    examples: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

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
                ColumnSpec('client_id','str','key',None,'unique client identifier'),
                ColumnSpec('segment','str','dimension',None,'client segment name'),
                ColumnSpec('revenue','float','metric','USD','gross revenue amount'),
                ColumnSpec('margin','float','metric','USD','gross margin amount'),
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
                ColumnSpec('client_id','str','key',None,'unique client identifier'),
                ColumnSpec('segment','str','dimension',None,'client segment name'),
                ColumnSpec('revenue','float','metric','USD','gross revenue amount'),
                ColumnSpec('margin','float','metric','USD','gross margin amount'),
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
        requires_columns=['client_id', 'revenue', 'margin'],
        guarantees_columns=['client_id','metric_value'],
        return_spec=ReturnSpec(
            grain='row per client',
            columns=[
                ColumnSpec('client_id','str','key',None,'unique client identifier'),
                ColumnSpec('metric_value','float','metric','USD','aggregated metric value for client'),
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
                ColumnSpec('client_id','str','key',None,'unique client identifier'),
                ColumnSpec('metric_value','float','metric','USD','metric used for ranking'),
                ColumnSpec('rank','int','dimension',None,'1 is best'),
            ]
        ),
        examples=['top5 = retrieve_top_clients(df, n=5, by="metric_value")'],
        tags=['rank','top-n']
    ),
]

STEP_INDEX: Dict[str, List[FunctionCard]] = {}
for fc in FUNCTION_CARDS:
    STEP_INDEX.setdefault(fc.step, []).append(fc)

# ------------------------------
# 2) Plan memory
# ------------------------------
@dataclass
class MemoryEntry:
    situation: str
    plan: str

PLAN_MEMORY: List[MemoryEntry] = [
    MemoryEntry('Top clients by revenue', 'load → filter → aggregate → rank'),
    MemoryEntry('Segment-level totals by margin', 'load → filter → aggregate'),
]

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|→")

def tokenize(text: str) -> set[str]:
    return set(w.lower() for w in re.findall(r"[A-Za-z0-9_]+|→", text))

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))

def retrieve_best_plan(question: str) -> str:
    qtok = tokenize(question)
    scored: List[Tuple[float,str]] = []
    for m in PLAN_MEMORY:
        scored.append((jaccard(qtok, tokenize(m.situation)), m.plan))
    scored.sort(reverse=True)
    return scored[0][1] if scored else 'load → filter → aggregate → rank'

# ------------------------------
# 3) Semantic grounding
# ------------------------------
@dataclass
class DataState:
    grain: str
    columns: Dict[str, ColumnSpec]

    @staticmethod
    def from_return_spec(rs: ReturnSpec) -> 'DataState':
        return DataState(
            grain=rs.grain,
            columns={c.name: c for c in rs.columns}
        )

    def ensure(self, needed: List[str]) -> List[str]:
        return [c for c in needed if c not in self.columns]

def pick_function_for_step(step: str, question: str, state: DataState) -> FunctionCard:
    candidates = STEP_INDEX.get(step, [])
    qtok = tokenize(question)

    def score(fc: FunctionCard) -> float:
        tag_score = len(set(fc.tags) & qtok)
        metric_hint = 1 if ('margin' in qtok and 'margin' in fc.tags) or ('revenue' in qtok and 'revenue' in fc.tags) else 0
        missing = state.ensure(fc.requires_columns)
        schema_penalty = len(missing)
        return tag_score + metric_hint - 0.5 * schema_penalty

    best = max(candidates, key=score) if candidates else None
    if not best:
        raise ValueError(f'No functions available for step {step}')
    return best

# ------------------------------
# 4) γ(3,4) graph builder
# ------------------------------
@dataclass
class Node:
    id: str
    type: Literal['event','thing','concept']
    label: str
    intentionality: float

@dataclass
class Edge:
    source: str
    target: str
    relation: Literal['NEAR','LEADS-TO','CONTAINS','EXPRESSES']

@dataclass
class Graph:
    nodes: List[Node]
    edges: List[Edge]

def build_gamma_graph(chain: List[FunctionCard]) -> Graph:
    nodes: List[Node] = []
    edges: List[Edge] = []
    def nid(label: str) -> str:
        return re.sub(r'[^a-z0-9_]+','', label.lower().replace(' ','_'))
    prev_id = None
    for fc in chain:
        fnode = Node(id=nid(fc.name), type='event', label=fc.name, intentionality=0.85)
        nodes.append(fnode)
        if prev_id:
            edges.append(Edge(source=prev_id, target=fnode.id, relation='LEADS-TO'))
        prev_id = fnode.id
        df_node_id = nid(fc.name + '_out_df')
        nodes.append(Node(id=df_node_id, type='thing', label=f"{fc.name} output df", intentionality=0.35))
        edges.append(Edge(source=fnode.id, target=df_node_id, relation='LEADS-TO'))
        for col in fc.return_spec.columns:
            col_id = nid(col.name)
            if all(n.id != col_id for n in nodes):
                nodes.append(Node(id=col_id, type='thing', label=col.name, intentionality=0.25))
            edges.append(Edge(source=df_node_id, target=col_id, relation='CONTAINS'))
        nodes.append(Node(id=nid(fc.step + '_concept'), type='concept', label=fc.step, intentionality=0.6))
        edges.append(Edge(source=fnode.id, target=nid(fc.step + '_concept'), relation='EXPRESSES'))
    return Graph(nodes=nodes, edges=edges)

# ------------------------------
# 5) Code synthesis
# ------------------------------
def synthesize_code(question: str) -> Tuple[str, Graph]:
    plan = retrieve_best_plan(question)
    steps = [s.strip() for s in plan.split('→')]
    selected: List[FunctionCard] = []
    start_date, end_date = '2024-01-01','2024-12-31'
    metric = 'margin' if 'margin' in question.lower() else 'revenue'
    topn = 5 if 'top' in question.lower() else 10
    state = DataState(grain='unknown', columns={})
    for step in steps:
        fc = pick_function_for_step(step, question, state)
        selected.append(fc)
        state = DataState.from_return_spec(fc.return_spec)
    imports = sorted({fc.name for fc in selected})
    lines = [f"from mypackage import {', '.join(imports)}", "", "# generated by reasoning_agent_mock"]
    var = 'df'
    for fc in selected:
        if fc.name == 'load_data':
            lines.append(f"{var} = load_data('{start_date}', '{end_date}')")
        elif fc.name == 'filter_clients':
            mr = 100.0 if metric == 'revenue' else 0.0
            lines.append(f"{var} = filter_clients({var}, min_revenue={mr})")
        elif fc.name == 'aggregate_by_metric':
            lines.append(f"{var} = aggregate_by_metric({var}, metric='{metric}', by='client_id')")
        elif fc.name == 'retrieve_top_clients':
            lines.append(f"top = retrieve_top_clients({var}, n={topn}, by='metric_value')")
            lines.append("print(top)")
    g = build_gamma_graph(selected)
    return "\n".join(lines), g

# ------------------------------
# Pretty-print graph
# ------------------------------
def format_graph(g: Graph) -> str:
    out = ["### Context: Retrieved Subgraph", "Nodes:"]
    for n in g.nodes:
        out.append(f"- {n.label} ({n.type}, intentionality={n.intentionality:.2f})")
    out.append("\nEdges:")
    for e in g.edges:
        out.append(f"- {e.source} -[{e.relation}]-> {e.target}")
    return "\n".join(out)

# ------------------------------
# Demo
# ------------------------------
if __name__ == '__main__':
    question = 'Top 5 clients by margin last year'
    question = "Segment-level totals by margin"
    code, graph = synthesize_code(question)
    print('--- Synthesized code ---')
    print(code)
    print('\n--- γ(3,4) graph ---')
    print(format_graph(graph))