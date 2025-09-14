# demo_clean.py
from sstt_agent import doc_to_n4l, N4LDuckDB, N4LRetriever

sme_doc = """
When retail flow is heavy, toxic takers reduce activity,
which causes market makers to narrow spreads.
Narrow spreads express confidence in liquidity.
"""

g = doc_to_n4l(sme_doc)
db = N4LDuckDB("n4l_clean.duckdb")
db.insert_graph(g, "doc1")

retr = N4LRetriever("n4l_clean.duckdb")
context, top_nodes, edges = retr.retrieve("Why do spreads narrow?", "doc1")

print(context)
