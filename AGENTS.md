# SSTorytime Agent Development Guide

## Build/Test/Lint Commands
- **Install/Setup**: `uv sync` to install dependencies
- **Lint**: `uv run ruff check` to check code quality, `uv run ruff check --fix` to auto-fix issues
- **Format**: `uv run ruff format` to format code  
- **Run basic demo**: `uv run python scripts/demo_clean.py`
- **Test domains**: `uv run python scripts/demo_software_engineering.py` (or medical_processes, research_methodology, agent_memory)
- **Validate design**: `uv run python scripts/validate_design_principles.py`
- **LLM integration**: `uv run python scripts/demo_llm_integration.py`
- **Document management**: `uv run python scripts/demo_document_management.py`
- **Module import test**: `uv run python -c "from sstt_agent import doc_to_n4l; print('OK')"`
- **No formal test suite**: No pytest/unittest setup - tests need to be added

## Architecture & Structure
- **Package**: `src/sstt_agent/` - Main Python package 
- **Core modules**: `pipeline.py` (doc_to_n4l), `storage_duckdb.py` (DuckDB), `retrieve.py` (query/context), `extract.py` (relation extraction)
- **Supporting**: `graph.py` (N4LGraph), `intent.py` (intentionality scoring), `canonical.py` (phrase formatting)
- **Database**: DuckDB for storing N4L graphs (`data/*.duckdb` files)
- **NLP**: Pattern-based extraction (no external models required)
- **Graph library**: NetworkX for graph operations
- **Scripts**: `scripts/` directory contains demos and examples

## SSTorytime Theoretical Foundation
Based on Mark Burgess's research papers in `/docs/`:
- **Semantic Spacetime**: Knowledge represented as stories/processes in spacetime rather than static ontologies
- **Intentionality**: Foreground concepts identified via frequency + work cost heuristics (fractionation theory)
- **Agent Semantics**: Context-aware reasoning where agents use graph traversal for understanding
- **N4L (Notes for Learning)**: Natural language input format that captures causal relationships
- **Graph as Process**: Graphs represent pathways through knowledge (stories) rather than just structures
- **Memory Integration**: Inspired by Memento (case-based retrieval) + ArcMemo (concept abstraction)

### Key Papers Context:
1. **Agent Semantics & Semantic Spacetime** (2506.07756v2): Foundational theory of graphs as spacetime processes
2. **Intentionality in Knowledge Representation** (2507.10000v1): Role of intentionality in scene context for cognitive agents  
3. **Memento** (2508.16153v2): Fine-tuning agents without LLM updates via memory-based learning
4. **ArcMemo** (2509.04439v2): Abstract reasoning with lifelong memory and concept-level abstraction

## Code Style & Conventions
- **Imports**: Standard library first, then third-party (networkx, duckdb), then local imports
- **Types**: Use type hints with `typing` module (Dict, List, Tuple, etc.)
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **String formatting**: Use f-strings preferentially  
- **Error handling**: Basic exception handling, no formal error classes yet
- **Database**: Use parameterized queries for DuckDB operations
- **Graph operations**: Use NetworkX DiGraph, store node types ("event", "thing", "concept") and edge types ("NEAR", "LEADS-TO", "CONTAINS", "EXPRESSES")
