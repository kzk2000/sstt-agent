#!/usr/bin/env python3
"""
Software Engineering Knowledge Demo
Demonstrates N4L extraction from software development practices
"""

from sstt_agent import doc_to_n4l, N4LDuckDB, N4LRetriever

# SME document about software engineering practices
software_doc = """
Code reviews reduce bugs by catching issues before deployment.
Automated testing increases confidence in releases while reducing manual effort.
Continuous integration catches integration issues early.
Documentation improves maintainability but requires ongoing effort to keep current.
Pair programming improves code quality by sharing knowledge between developers.
Technical debt accumulates when shortcuts are taken to meet deadlines.
Refactoring reduces technical debt but requires dedicated time allocation.
"""


def run_software_demo():
    print("=== Software Engineering Knowledge Demo ===\n")

    # Convert SME document to N4L graph
    print("📄 Processing software engineering document...")
    g = doc_to_n4l(software_doc)
    print(f"   Graph: {len(g.g.nodes)} nodes, {len(g.g.edges)} edges")

    # Store in DuckDB
    print("\n💾 Storing in DuckDB...")
    db = N4LDuckDB("software_knowledge.duckdb")
    db.insert_graph(g, "software_practices")

    # Test different queries
    retr = N4LRetriever("software_knowledge.duckdb")

    queries = [
        "How to reduce bugs in software?",
        "What improves code quality?",
        "How to manage technical debt?",
        "Benefits of automated testing?",
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        context, top_nodes, edges = retr.retrieve(query, "software_practices")
        print(context)
        print(f"   Top nodes: {[n[0] for n in top_nodes[:2]]}")


if __name__ == "__main__":
    run_software_demo()
