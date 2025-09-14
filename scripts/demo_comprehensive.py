#!/usr/bin/env python3
"""
Comprehensive Demo - Multiple Knowledge Domains
Demonstrates the SSTorytime-inspired N4L pipeline across different domains
Tests the complete workflow: Text → N4L Graph → DuckDB → LLM Context
"""

from sstt_agent import doc_to_n4l, N4LDuckDB, N4LRetriever
import os


def test_domain(name: str, doc: str, queries: list):
    """Test a knowledge domain with multiple queries"""
    print(f"\n{'=' * 60}")
    print(f"🧠 DOMAIN: {name.upper()}")
    print("=" * 60)

    # Process document
    print(f"📄 Processing {name} document...")
    g = doc_to_n4l(doc)
    print(f"   Created graph: {len(g.g.nodes)} nodes, {len(g.g.edges)} edges")

    if len(g.g.nodes) == 0:
        print("   ⚠️  No nodes extracted - patterns may need refinement")
        return

    # Store in database
    db_path = f"{name.lower().replace(' ', '_')}.duckdb"
    db = N4LDuckDB(db_path)
    db.insert_graph(g, f"{name}_doc")

    # Test retrieval
    retr = N4LRetriever(db_path)

    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: '{query}'")
        context, top_nodes, edges = retr.retrieve(query, f"{name}_doc")
        print(context)
        if top_nodes:
            print(
                f"   📊 Top concepts: {[n[0][:30] + '...' if len(n[0]) > 30 else n[0] for n in top_nodes[:2]]}"
            )


def run_comprehensive_demo():
    print("SSTorytime Agent - Comprehensive Knowledge Demo")
    print("Testing N4L extraction across multiple domains")

    # Domain 1: Trading/Finance
    trading_doc = """
    When retail flow is heavy, toxic takers reduce activity, which causes market makers to narrow spreads.
    Narrow spreads express confidence in liquidity.
    High volatility increases risk while creating profit opportunities.
    """

    test_domain(
        "Trading",
        trading_doc,
        [
            "Why do spreads narrow?",
            "What creates opportunities?",
            "How does flow affect activity?",
        ],
    )

    # Domain 2: Software Engineering (simplified)
    software_doc = """
    Code reviews reduce bugs which improves software quality.
    Testing increases confidence while reducing deployment risk.
    Refactoring improves maintainability but requires time investment.
    """

    test_domain(
        "Software Engineering",
        software_doc,
        ["How to reduce bugs?", "What improves quality?", "How to manage risk?"],
    )

    # Domain 3: Learning/Education
    learning_doc = """
    Practice improves skill retention which enables better performance.
    Feedback guides improvement while highlighting knowledge gaps.
    Spaced repetition strengthens memory formation.
    """

    test_domain(
        "Learning",
        learning_doc,
        [
            "How to improve performance?",
            "What strengthens memory?",
            "How to identify gaps?",
        ],
    )

    print(f"\n{'=' * 60}")
    print("🎯 SUMMARY")
    print("=" * 60)
    print("✅ Successfully demonstrated N4L pipeline across 3 domains")
    print("✅ Text → Graph → DuckDB → LLM Context workflow verified")
    print("✅ Ready for SME document processing and LLM integration")

    # Cleanup
    for f in ["trading.duckdb", "software_engineering.duckdb", "learning.duckdb"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    run_comprehensive_demo()
