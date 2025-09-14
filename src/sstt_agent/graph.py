import networkx as nx


class N4LGraph:
    def __init__(self):
        self.g = nx.DiGraph()

    def add_node(self, label: str, node_type: str, intent: float):
        if label not in self.g:
            self.g.add_node(label, type=node_type, intent=float(intent))

    def add_edge(self, src: str, edge_type: str, dst: str):
        self.g.add_edge(src, dst, type=edge_type)

    def to_dict(self):
        return {
            "nodes": [{"label": n, **self.g.nodes[n]} for n in self.g.nodes],
            "edges": [
                {"src": u, "dst": v, **self.g.edges[u, v]} for u, v in self.g.edges
            ],
        }
