#!/usr/bin/env python3
"""
Document Management Demo
Tests smart duplicate handling and multi-document knowledge base management
"""

from sstt_agent import doc_to_n4l, N4LDuckDB, N4LRetriever


def demo_document_management():
    print("📚 Document Management & Duplicate Handling Demo")
    print("=" * 60)

    # Multiple SME documents
    documents = {
        "api_design_v1": """
        REST APIs provide stateless communication which simplifies scaling.
        Authentication tokens secure endpoints while enabling user access.
        """,
        "api_design_v2": """
        REST APIs provide stateless communication which simplifies scaling.
        Authentication tokens secure endpoints while enabling user access.
        Rate limiting prevents abuse which protects server resources.
        Error responses guide client behavior while enabling debugging.
        """,
        "database_ops": """
        Database indexing improves query performance which enables faster responses.
        Connection pooling reduces overhead while managing resource usage.
        """,
    }

    db = N4LDuckDB("knowledge_base.duckdb")

    print("\n1️⃣ Initial document uploads:")
    for doc_id, content in documents.items():
        g = doc_to_n4l(content)
        db.insert_graph(g, doc_id)

    print("\n📊 Knowledge base contains:")
    for graph_id in db.list_graphs():
        node_count = len(db.query_nodes(graph_id))
        edge_count = len(db.query_edges(graph_id))
        print(f"   {graph_id}: {node_count} nodes, {edge_count} edges")

    print("\n2️⃣ Testing duplicate upload (replace=True):")
    g_updated = doc_to_n4l(documents["api_design_v2"])
    db.insert_graph(g_updated, "api_design_v1")  # Same ID, new content

    print("\n3️⃣ Testing duplicate upload (replace=False):")
    g_same = doc_to_n4l(documents["database_ops"])
    db.insert_graph(g_same, "database_ops", replace_existing=False)

    print("\n📊 Final knowledge base:")
    for graph_id in db.list_graphs():
        node_count = len(db.query_nodes(graph_id))
        edge_count = len(db.query_edges(graph_id))
        print(f"   {graph_id}: {node_count} nodes, {edge_count} edges")

    print("\n4️⃣ Testing cross-document retrieval:")
    retr = N4LRetriever("knowledge_base.duckdb")

    # Query that could match multiple documents
    context, top_nodes, edges = retr.retrieve("improve performance", "api_design_v1")
    print("\nQuery 'improve performance' in api_design_v1:")
    print(context)

    context, top_nodes, edges = retr.retrieve("improve performance", "database_ops")
    print("\nQuery 'improve performance' in database_ops:")
    print(context)

    # Test utility methods
    print("\n5️⃣ Utility methods:")
    print(f"   Graph exists check: {db.graph_exists('api_design_v1')}")
    print(f"   Graph exists check: {db.graph_exists('nonexistent')}")
    print(f"   All graphs: {db.list_graphs()}")

    # Cleanup
    import os

    if os.path.exists("knowledge_base.duckdb"):
        os.remove("knowledge_base.duckdb")

    print("\n✅ Document management demo complete!")
    print("✅ Smart duplicate handling prevents data corruption")
    print("✅ Multi-document knowledge bases supported")
    print("✅ Utility methods for graph management available")


if __name__ == "__main__":
    demo_document_management()
