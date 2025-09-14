#!/usr/bin/env python3
"""
Agent Memory Knowledge Demo
Demonstrates N4L extraction for AI agent memory systems
Inspired by the Memento paper and SSTorytime's agent semantics concepts
"""

from sstt_agent import doc_to_n4l, N4LDuckDB, N4LRetriever

# SME document about agent memory and cognitive processes
agent_memory_doc = """
Working memory maintains current task context while filtering irrelevant information.
Long-term memory stores experiences that influence future decision-making patterns.
Episodic memory captures specific events that provide contextual understanding.
Semantic memory represents abstract knowledge that generalizes across situations.
Memory consolidation strengthens important connections while weakening unused pathways.
Retrieval cues trigger relevant memories that inform current reasoning processes.
Memory interference occurs when similar experiences conflict during recall.
Attention mechanisms select relevant information while suppressing distracting stimuli.
Context switching requires cognitive resources that temporarily reduce performance.
Memory decay affects older information that lacks reinforcement through repeated access.
"""


def run_agent_memory_demo():
    print("=== Agent Memory Knowledge Demo ===\n")

    # Convert SME document to N4L graph
    print("🧠 Processing agent memory document...")
    g = doc_to_n4l(agent_memory_doc)
    print(f"   Graph: {len(g.g.nodes)} nodes, {len(g.g.edges)} edges")

    # Store in DuckDB
    print("\n💾 Storing in DuckDB...")
    db = N4LDuckDB("agent_memory.duckdb")
    db.insert_graph(g, "memory_systems")

    # Test different memory-related queries
    retr = N4LRetriever("agent_memory.duckdb")

    queries = [
        "How does memory affect reasoning?",
        "What influences decision-making?",
        "How to maintain task context?",
        "What causes memory interference?",
        "How does attention work?",
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        context, top_nodes, edges = retr.retrieve(query, "memory_systems")
        print(context)
        print(f"   Top nodes: {[n[0] for n in top_nodes[:2]]}")


if __name__ == "__main__":
    run_agent_memory_demo()
