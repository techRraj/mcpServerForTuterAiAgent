# MCP RAG + Agent Server for AI Tutor

A complete Model Context Protocol (MCP) server that combines:
- **RAG** (Retrieval-Augmented Generation) with PostgreSQL + pgvector
- **Free LLM** integration via OpenRouter
- **Dynamic agent creation** as an MCP tool
- **FastMCP** framework for building the server

## Features

### Tools
- `ingest_documents` — Import documents into the vector knowledge base
- `search_knowledge` — Semantic search using cosine similarity
- `ask_with_rag` — Full RAG pipeline (retrieve + LLM generation)
- `create_agent` — Dynamically create and run specialized AI agents

### Resources
- `knowledge://stats` — Knowledge base statistics
- `knowledge://config` — Current server configuration

### Prompts
- `explain_concept` — Generate explanation prompts for any concept

## Architecture
