import duckdb
import os
from .graph import N4LGraph


class N4LDuckDB:
    def __init__(self, db_path="n4l.duckdb"):
        # Ensure data directory exists and store databases there
        if not os.path.isabs(db_path):
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, db_path)
        
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
          graph_id VARCHAR,
          label    VARCHAR,
          node_type VARCHAR,
          intent_score DOUBLE
        )""")
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
          graph_id VARCHAR,
          src      VARCHAR,
          dst      VARCHAR,
          edge_type VARCHAR
        )""")

    def insert_graph(self, graph: N4LGraph, graph_id: str, replace_existing=True):
        """Insert graph with smart duplicate handling.

        Args:
            graph: N4LGraph to insert
            graph_id: Unique identifier for this graph
            replace_existing: If True, replace existing graph_id; if False, skip if exists
        """
        # Check if graph_id already exists
        existing_nodes = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE graph_id=?", [graph_id]
        ).fetchone()[0]

        if existing_nodes > 0:
            if replace_existing:
                print(
                    f"⚠️  Replacing existing graph '{graph_id}' ({existing_nodes} nodes)"
                )
                self._delete_graph(graph_id)
            else:
                print(f"⏭️  Graph '{graph_id}' already exists, skipping insert")
                return

        # Insert new graph data
        nodes = [
            (graph_id, n, d["type"], d["intent"]) for n, d in graph.g.nodes(data=True)
        ]
        edges = [(graph_id, u, v, d["type"]) for u, v, d in graph.g.edges(data=True)]

        if nodes:
            self.conn.executemany("INSERT INTO nodes VALUES (?,?,?,?)", nodes)
        if edges:
            self.conn.executemany("INSERT INTO edges VALUES (?,?,?,?)", edges)

        print(f"✅ Inserted graph '{graph_id}': {len(nodes)} nodes, {len(edges)} edges")

    def _delete_graph(self, graph_id: str):
        """Delete all nodes and edges for a graph_id"""
        self.conn.execute("DELETE FROM nodes WHERE graph_id=?", [graph_id])
        self.conn.execute("DELETE FROM edges WHERE graph_id=?", [graph_id])

    def query_nodes(self, graph_id: str):
        return self.conn.execute(
            "SELECT label,node_type,intent_score FROM nodes WHERE graph_id=?",
            [graph_id],
        ).fetchall()

    def query_edges(self, graph_id: str):
        return self.conn.execute(
            "SELECT src,dst,edge_type FROM edges WHERE graph_id=?", [graph_id]
        ).fetchall()

    def list_graphs(self):
        """List all graph_ids in the database"""
        return [
            row[0]
            for row in self.conn.execute(
                "SELECT DISTINCT graph_id FROM nodes ORDER BY graph_id"
            ).fetchall()
        ]

    def graph_exists(self, graph_id: str) -> bool:
        """Check if a graph_id already exists"""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE graph_id=?", [graph_id]
        ).fetchone()[0]
        return count > 0

    def close(self):
        self.conn.close()
