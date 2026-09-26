"""MCP RAG + Agent server for VS Code + Copilot Chat."""
import sys
import logging
from fastmcp import FastMCP
from db import init_db, insert_document, insert_embedding, search_similar, get_stats
from llm_client import get_embedding, rag_query
from agent_executor import create_and_run_agent
from typing import Optional

# Log to stderr — stdout is reserved for MCP protocol under stdio transport
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stderr)
logger = logging.getLogger("MCP-RAG-Server")

mcp = FastMCP("AI_Tutor_RAG",
                instructions="RAG + Agent creation server. Use ingest_documents "
                             "to load knowledge, then search_knowledge or ask_with_rag.")

init_db()


@mcp.tool()
def ingest_documents(documents: list[dict]) -> dict:
    """Import documents. Each needs 'content'; 'source' and 'metadata' optional."""
    imported, errors = 0, []
    for doc in documents:
        try:
            content = doc.get("content", "").strip()
            if not content:
                continue
            doc_id = insert_document(
                content=content,
                source=doc.get("source", "manual"),
                metadata=doc.get("metadata", {})
            )
            insert_embedding(doc_id, get_embedding(content))
            imported += 1
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Failed to ingest: {e}")
    return {"imported": imported, "errors": errors, "total_attempted": len(documents)}


@mcp.tool()
def search_knowledge(query: str, top_k: int = 5) -> dict:
    """Semantic search over the knowledge base. Returns docs + similarity scores."""
    results = search_similar(get_embedding(query), top_k=top_k)
    return {"query": query, "results_count": len(results), "results": results}


@mcp.tool()
def ask_with_rag(question: str, top_k: int = 5) -> dict:
    """Full RAG: retrieve context, then LLM answers grounded in it."""
    docs = search_similar(get_embedding(question), top_k=top_k)
    if not docs:
        return {"question": question,
                "answer": "No relevant info. Use ingest_documents first.",
                "sources": []}
    answer = rag_query(question, docs)
    return {"question": question, "answer": answer,
            "sources": [{"id": d["id"], "source": d["source"],
                         "similarity": d["similarity"]} for d in docs]}


@mcp.tool()
def create_agent(agent_name: str, system_prompt: str, task: str,
                 model: Optional[str] = None) -> dict:
    """
    Dynamically create an agent and execute a task.

    Args:
        agent_name: Identifier for the agent.
        system_prompt: System prompt defining the agent's role.
        task: The concrete task to perform.
        model: Optional model override.
    """
    logger.info(f"Creating agent: {agent_name}")
    return create_and_run_agent(agent_name, system_prompt, task, model)


@mcp.resource("knowledge://stats")
def knowledge_stats() -> dict:
    """Knowledge base statistics."""
    return get_stats()


@mcp.resource("knowledge://config")
def knowledge_config() -> dict:
    """Current server configuration."""
    import os
    return {
        "chat_model": os.getenv("OPENROUTER_MODEL"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "embedding_base_url": os.getenv("EMBEDDING_BASE_URL"),
        "vector_db": "PostgreSQL + pgvector",
        "rag_enabled": True,
        "agent_creation_enabled": True,
    }


@mcp.prompt()
def explain_concept(concept: str) -> str:
    """Generate a prompt for explaining a concept."""
    return f"Explain '{concept}' in simple terms, assuming I'm new to MCP and AI agents."


if __name__ == "__main__":
    # stdio transport for VS Code Copilot Chat
    mcp.run()