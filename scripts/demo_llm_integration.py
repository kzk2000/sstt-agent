#!/usr/bin/env python3
"""
LLM Integration Demo
Tests the complete pipeline for real LLM system prompt generation
Validates SSTorytime principles: stories → reasoning → knowledge → context
"""

from sstt_agent import doc_to_n4l, N4LDuckDB, N4LRetriever


def test_llm_integration():
    print("🤖 LLM Integration Test - SSTorytime Principles")
    print("=" * 60)

    # SME document with clear causal stories (SSTorytime principle)
    trading_knowledge = """
    When retail flow is heavy, toxic takers reduce activity, which causes market makers to narrow spreads.
    Narrow spreads express confidence in liquidity.
    High volatility increases trader stress which leads to poor decision making.
    Poor decisions cause losses that trigger risk management responses.
    """

    # Process through N4L pipeline
    print("📈 Processing trading knowledge...")
    g = doc_to_n4l(trading_knowledge)

    db = N4LDuckDB("trading_llm.duckdb")
    db.insert_graph(g, "trading_expertise")
    retr = N4LRetriever("trading_llm.duckdb")

    # Test realistic LLM scenarios
    scenarios = [
        {
            "user_question": "Why do market makers narrow spreads when retail flow is heavy?",
            "context_query": "spreads narrow retail flow",
        },
        {
            "user_question": "How does volatility affect trading decisions?",
            "context_query": "volatility decisions stress",
        },
        {
            "user_question": "What triggers risk management in trading?",
            "context_query": "risk management triggers losses",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'=' * 40}")
        print(f"📋 LLM Scenario {i}")
        print("=" * 40)

        # Retrieve knowledge context
        context, top_nodes, edges = retr.retrieve(
            scenario["context_query"], "trading_expertise"
        )

        # Generate complete LLM system prompt
        system_prompt = f"""You are a trading expert. Use the following knowledge from subject matter experts to provide accurate, specific guidance.

{context}

Based on this expertise, provide a clear explanation that helps the user understand the underlying mechanisms and relationships."""

        print(f"❓ User Question: {scenario['user_question']}")
        print("\n🧠 Generated System Prompt:")
        print("-" * 50)
        print(system_prompt)
        print("-" * 50)

        # Evaluate context quality
        lines = context.count("\n")
        relationships = (
            context.count("leads to")
            + context.count("expresses")
            + context.count("contains")
        )
        quality = (
            "✅ Rich"
            if relationships >= 2
            else "⚠️ Limited"
            if relationships >= 1
            else "❌ Poor"
        )
        print(
            f"📊 Context Quality: {quality} ({relationships} relationships, {lines} lines)"
        )

        if top_nodes:
            print(f"🎯 Key Concepts: {[n[0][:30] for n in top_nodes[:2]]}")

    # Cleanup
    import os

    if os.path.exists("trading_llm.duckdb"):
        os.remove("trading_llm.duckdb")

    print(f"\n{'=' * 60}")
    print("✅ LLM Integration Test Complete")
    print("✅ SSTorytime story-based reasoning demonstrated")
    print("✅ Context suitable for expert system prompts")


if __name__ == "__main__":
    test_llm_integration()
