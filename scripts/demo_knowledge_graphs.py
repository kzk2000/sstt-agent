#!/usr/bin/env python3
"""
Knowledge Graph Principles Demo
Demonstrates N4L extraction for knowledge representation concepts
Directly inspired by SSTorytime's semantic spacetime and graph reasoning principles
"""

from sstt_agent import doc_to_n4l, N4LDuckDB, N4LRetriever

# SME document about knowledge graphs and semantic representation
knowledge_graph_doc = """
Semantic relationships connect concepts through meaningful associations.
Graph topology reveals structural patterns that indicate information centrality.
Node centrality measures importance based on connection density and traversal frequency.
Edge weights represent relationship strength while enabling weighted path calculations.
Knowledge provenance tracks information sources to establish credibility and context.
Temporal relationships capture sequence dependencies that influence causal reasoning.
Contextual embeddings enhance retrieval by incorporating situational relevance.
Graph traversal algorithms discover indirect connections between seemingly unrelated concepts.
Ontology alignment enables knowledge fusion across different domain representations.
Intentionality scoring prioritizes foreground concepts while filtering background noise.
"""


def run_knowledge_graph_demo():
    print("=== Knowledge Graph Principles Demo ===\n")

    # Convert SME document to N4L graph
    print("🕸️  Processing knowledge graph document...")
    g = doc_to_n4l(knowledge_graph_doc)
    print(f"   Graph: {len(g.g.nodes)} nodes, {len(g.g.edges)} edges")

    # Store in DuckDB
    print("\n💾 Storing in DuckDB...")
    db = N4LDuckDB("knowledge_graphs.duckdb")
    db.insert_graph(g, "graph_principles")

    # Test different knowledge graph queries
    retr = N4LRetriever("knowledge_graphs.duckdb")

    queries = [
        "How to measure concept importance?",
        "What enables knowledge fusion?",
        "How to discover hidden connections?",
        "What tracks information sources?",
        "How to filter noise from knowledge?",
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        context, top_nodes, edges = retr.retrieve(query, "graph_principles")
        print(context)
        print(f"   Top nodes: {[n[0] for n in top_nodes[:2]]}")


if __name__ == "__main__":
    run_knowledge_graph_demo()
