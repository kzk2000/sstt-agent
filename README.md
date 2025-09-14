# SSTorytime Agent

A Python implementation inspired by Mark Burgess's SSTorytime project for extracting knowledge graphs from subject matter expert (SME) documents and retrieving contextual information for LLM prompting. Incorporates insights from Memento and ArcMemo papers on agent memory systems.

## Quick Start

```bash
# Install dependencies  
uv sync

# Run basic demo
uv run python scripts/demo_clean.py

# Test different domains
uv run python scripts/demo_software_engineering.py
uv run python scripts/demo_medical_processes.py

# Validate design principles
uv run python scripts/validate_design_principles.py

# Test import
uv run python -c "from sstt_agent import doc_to_n4l; print('OK')"
```

## How it Works

1. **Text → N4L Graph**: Extracts semantic relationships (stories/processes) from SME documents
2. **Intentionality Scoring**: Ranks concepts by importance using Burgess's fractionation heuristics
3. **Graph → DuckDB**: Stores nodes (events/things/concepts) and edges (LEADS-TO, EXPRESSES, CONTAINS, NEAR) in `data/` directory
4. **Query → Context**: Retrieves relevant subgraph as LLM-ready expert context

## Example Usage

```python
from sstt_agent import doc_to_n4l, N4LDuckDB, N4LRetriever

# Load SME knowledge
doc = """When retail flow is heavy, toxic takers reduce activity, 
which causes market makers to narrow spreads. 
Narrow spreads express confidence in liquidity."""

g = doc_to_n4l(doc)

# Store in database (smart duplicate handling)
db = N4LDuckDB("knowledge.duckdb")
db.insert_graph(g, "trading_doc")  # Replaces existing by default
# db.insert_graph(g, "trading_doc", replace_existing=False)  # Skip if exists

# Retrieve for LLM context
retr = N4LRetriever("knowledge.duckdb") 
context, top_nodes, edges = retr.retrieve("Why do spreads narrow?", "trading_doc")

# Use in LLM system prompt
system_prompt = f"""You are a trading expert. Use this knowledge:

{context}

Answer the user's question based on this expertise."""
```

## Demo Scripts

- `demo_clean.py` - Basic trading/finance knowledge
- `demo_software_engineering.py` - Software development practices  
- `demo_medical_processes.py` - Healthcare domain expertise
- `demo_research_methodology.py` - Academic research processes
- `demo_agent_memory.py` - AI agent memory systems
- `demo_llm_integration.py` - Complete LLM workflow
- `demo_enhanced_memory.py` - Cross-domain reasoning
- `demo_document_management.py` - Smart duplicate handling  
- `validate_design_principles.py` - Comprehensive validation

## Architecture

```
src/sstt_agent/
├── pipeline.py          # Main doc_to_n4l function
├── extract.py           # Pattern-based relation extraction
├── storage_duckdb.py    # DuckDB persistence layer
├── retrieve.py          # Query & LLM context formatting
├── graph.py             # N4LGraph NetworkX wrapper
├── intent.py            # Intentionality scoring (Burgess)
└── canonical.py         # Phrase canonicalization
```

## Theoretical Foundation

Based on Mark Burgess's SSTorytime research on semantic spacetime and knowledge graphs:
- **Stories as Process**: Knowledge represented as causal narratives rather than static facts
- **Intentionality**: Foreground concepts identified through frequency and work cost heuristics  
- **Graph Reasoning**: Semantic relationships enable traversal and story reconstruction
- **Agent Semantics**: Context-aware knowledge retrieval for intelligent systems