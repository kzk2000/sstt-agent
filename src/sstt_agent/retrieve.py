from difflib import SequenceMatcher
import duckdb
import os


class N4LRetriever:
    def __init__(self, db_path="n4l.duckdb"):
        # Use data directory for relative paths
        if not os.path.isabs(db_path):
            db_path = os.path.join("data", db_path)
        self.conn = duckdb.connect(db_path)

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def retrieve(self, query: str, graph_id: str, top_k=3, intent_weight=0.6):
        nodes = self.conn.execute(
            "SELECT label,node_type,intent_score FROM nodes WHERE graph_id=?",
            [graph_id],
        ).fetchall()
        edges = self.conn.execute(
            "SELECT src,dst,edge_type FROM edges WHERE graph_id=?", [graph_id]
        ).fetchall()

        scored = []
        for label, ntype, intent in nodes:
            sim = self._similarity(query, label)
            score = sim + intent_weight * float(intent)
            scored.append((label, ntype, float(intent), score))

        top = sorted(scored, key=lambda x: x[3], reverse=True)[:top_k]
        seeds = {t[0] for t in top}

        sub_edges = [e for e in edges if e[0] in seeds or e[1] in seeds]

        seen = set()
        clean_edges = []
        for src, dst, et in sub_edges:
            k = (src, et, dst)
            if k not in seen:
                clean_edges.append((src, dst, et))
                seen.add(k)

        # Filter out messy/redundant phrases for cleaner LLM context
        filtered_edges = []
        for src, dst, et in clean_edges:
            # Skip overly long or redundant phrases
            if len(src) > 60 or len(dst) > 60:
                continue
            if "Causing" in src and ("Cause" in src or "Which" in src):
                continue
            filtered_edges.append((src, dst, et))

        lines = []
        for src, dst, et in filtered_edges:
            if et == "LEADS-TO":
                lines.append(f"- {src} leads to {dst}.")
            elif et == "EXPRESSES":
                lines.append(f"- {src} expresses {dst}.")
            elif et == "CONTAINS":
                lines.append(f"- {src} contains {dst}.")
            elif et == "NEAR":
                lines.append(f"- {src} is related to {dst}.")

        # Create LLM-suitable context
        if lines:
            context = (
                "### Retrieved Knowledge Context\n\nKey relationships from the document:\n"
                + "\n".join(lines)
                + "\n\nUse this context to answer questions about the subject matter."
            )
        else:
            context = "### Retrieved Knowledge Context\n\nNo clear relationships found for this query."
        return context, top, clean_edges
