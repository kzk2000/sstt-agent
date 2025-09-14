#!/usr/bin/env python3
"""
Medical Process Knowledge Demo
Demonstrates N4L extraction from healthcare domain expertise
Based on SSTorytime principles of process representation and intentionality
"""

from sstt_agent import doc_to_n4l, N4LDuckDB, N4LRetriever

# SME document about medical diagnosis and treatment processes
medical_doc = """
Patient symptoms guide initial diagnostic assessments.
Laboratory tests confirm or rule out suspected conditions.
Imaging studies reveal structural abnormalities that clinical examination cannot detect.
Differential diagnosis narrows possible conditions through systematic elimination.
Treatment protocols reduce symptom severity while minimizing adverse effects.
Patient monitoring tracks treatment response and identifies complications early.
Drug interactions increase risk of adverse events when multiple medications are prescribed.
Preventive care reduces disease incidence through early intervention strategies.
Medical history provides context that influences diagnostic accuracy.
"""


def run_medical_demo():
    print("=== Medical Process Knowledge Demo ===\n")

    # Convert SME document to N4L graph
    print("🏥 Processing medical process document...")
    g = doc_to_n4l(medical_doc)
    print(f"   Graph: {len(g.g.nodes)} nodes, {len(g.g.edges)} edges")

    # Store in DuckDB
    print("\n💾 Storing in DuckDB...")
    db = N4LDuckDB("medical_knowledge.duckdb")
    db.insert_graph(g, "medical_processes")

    # Test different medical queries
    retr = N4LRetriever("medical_knowledge.duckdb")

    queries = [
        "How are conditions diagnosed?",
        "What reduces disease risk?",
        "How to monitor treatment effectiveness?",
        "What increases adverse events?",
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        context, top_nodes, edges = retr.retrieve(query, "medical_processes")
        print(context)
        print(f"   Top nodes: {[n[0] for n in top_nodes[:2]]}")


if __name__ == "__main__":
    run_medical_demo()
