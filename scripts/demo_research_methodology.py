#!/usr/bin/env python3
"""
Research Methodology Knowledge Demo
Demonstrates N4L extraction for academic research processes
Inspired by SSTorytime's focus on stories, reasoning, and knowledge representation
"""

from sstt_agent import doc_to_n4l, N4LDuckDB, N4LRetriever

# SME document about research methodology and academic processes
research_doc = """
Literature reviews establish foundational knowledge while identifying research gaps.
Hypothesis formation guides experimental design and methodology selection.
Data collection requires careful sampling to ensure representative results.
Statistical analysis reveals patterns that support or refute hypotheses.
Peer review improves research quality through expert evaluation and feedback.
Replication studies increase confidence in findings by validating results across contexts.
Publication bias affects knowledge advancement when negative results remain unpublished.
Interdisciplinary collaboration expands research scope while introducing methodological challenges.
Research ethics protects participants while ensuring scientific integrity.
Grant funding enables research projects but influences topic selection priorities.
"""


def run_research_demo():
    print("=== Research Methodology Knowledge Demo ===\n")

    # Convert SME document to N4L graph
    print("📚 Processing research methodology document...")
    g = doc_to_n4l(research_doc)
    print(f"   Graph: {len(g.g.nodes)} nodes, {len(g.g.edges)} edges")

    # Store in DuckDB
    print("\n💾 Storing in DuckDB...")
    db = N4LDuckDB("research_knowledge.duckdb")
    db.insert_graph(g, "research_methods")

    # Test different research queries
    retr = N4LRetriever("research_knowledge.duckdb")

    queries = [
        "How to establish research foundations?",
        "What improves research quality?",
        "How to ensure valid results?",
        "What affects knowledge advancement?",
        "How to design experiments?",
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        context, top_nodes, edges = retr.retrieve(query, "research_methods")
        print(context)
        print(f"   Top nodes: {[n[0] for n in top_nodes[:2]]}")


if __name__ == "__main__":
    run_research_demo()
