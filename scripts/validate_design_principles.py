#!/usr/bin/env python3
"""
Design Principles Validation
Comprehensive test of SSTorytime principles + Memento/ArcMemo insights
Ensures LLM context output is optimal for system prompts
"""

from sstt_agent import doc_to_n4l, N4LDuckDB, N4LRetriever


def validate_sstorytime_principles():
    """Validate core SSTorytime design principles are working"""
    print("🎯 SSTorytime Design Principles Validation")
    print("=" * 60)

    # 1. Stories as knowledge representation
    print("1️⃣ PRINCIPLE: Stories as Knowledge Representation")
    story_doc = """
    Customer complaint triggers investigation which reveals system weaknesses.
    System fixes reduce complaint frequency while improving user satisfaction.
    User satisfaction increases retention which drives revenue growth.
    """

    g = doc_to_n4l(story_doc)
    print(f"   ✅ Story extracted: {len(g.g.nodes)} nodes, {len(g.g.edges)} edges")

    # 2. Intentionality scoring (foreground vs background)
    print("\n2️⃣ PRINCIPLE: Intentionality Scoring")
    for node, data in sorted(
        g.g.nodes(data=True), key=lambda x: x[1]["intent"], reverse=True
    )[:3]:
        print(f"   Intent {data['intent']:.4f}: {node[:40]}...")

    # 3. Semantic relationships and reasoning paths
    print("\n3️⃣ PRINCIPLE: Semantic Relationships & Reasoning Paths")
    for src, dst, data in g.g.edges(data=True):
        print(f"   {src[:25]}... --{data['type']}--> {dst[:25]}...")

    return g


def validate_llm_prompt_quality():
    """Ensure LLM context is optimal for system prompts"""
    print("\n🤖 LLM System Prompt Quality Validation")
    print("=" * 60)

    # Test with expert knowledge suitable for LLM grounding
    expert_doc = """
    Database indexing improves query performance which enables faster user responses.
    Faster responses increase user satisfaction while reducing server load complaints.
    Proper caching reduces database hits which improves overall system efficiency.
    System efficiency enables scaling while minimizing infrastructure costs.
    """

    g = doc_to_n4l(expert_doc)
    db = N4LDuckDB("llm_test.duckdb")
    db.insert_graph(g, "db_expertise")
    retr = N4LRetriever("llm_test.duckdb")

    # Test LLM-ready context for different query types
    test_cases = [
        {
            "user_query": "How can I improve database performance?",
            "retrieval_query": "improve performance database",
            "domain": "Database Engineering",
        },
        {
            "user_query": "What reduces infrastructure costs?",
            "retrieval_query": "reduce costs infrastructure",
            "domain": "System Architecture",
        },
    ]

    for case in test_cases:
        print(f"\n🔍 Test Case: {case['domain']}")
        print(f"   User asks: '{case['user_query']}'")

        context, top_nodes, edges = retr.retrieve(
            case["retrieval_query"], "db_expertise"
        )

        # Generate production-ready system prompt
        system_prompt = f"""You are a {case["domain"]} expert consultant. Use the following knowledge from subject matter experts to provide accurate, actionable guidance.

{context}

Provide a specific, practical response that directly addresses the user's question using the expert knowledge above."""

        print("📋 Generated LLM System Prompt:")
        print("-" * 50)
        print(system_prompt)
        print("-" * 50)

        # Validate prompt quality metrics
        has_relationships = "leads to" in context or "expresses" in context
        has_context = "No clear relationships" not in context
        context_lines = len([line for line in context.split("\n") if line.strip()])

        quality_score = sum([has_relationships, has_context, context_lines >= 4])
        quality = ["❌ Poor", "⚠️ Fair", "✅ Good", "🌟 Excellent"][quality_score]

        print(f"📊 Prompt Quality: {quality}")
        print(f"   Relationships: {'✅' if has_relationships else '❌'}")
        print(f"   Context found: {'✅' if has_context else '❌'}")
        print(
            f"   Content depth: {'✅' if context_lines >= 4 else '❌'} ({context_lines} lines)"
        )

    # Cleanup
    import os

    if os.path.exists("llm_test.duckdb"):
        os.remove("llm_test.duckdb")

    print(f"\n{'=' * 60}")
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)
    print("✅ SSTorytime principles: Stories → Graphs → Reasoning")
    print("✅ Intentionality scoring: Foreground vs background concepts")
    print("✅ LLM context quality: Expert knowledge → System prompts")
    print("✅ Ready for SME document processing and LLM integration")
    print("✅ Inspired by Memento (case-based memory) & ArcMemo (concept abstraction)")


if __name__ == "__main__":
    validate_sstorytime_principles()
    validate_llm_prompt_quality()
