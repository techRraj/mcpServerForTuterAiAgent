"""Free LLM wrapper using OpenRouter (OpenAI-compatible)."""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenRouter uses the OpenAI SDK with a custom base_url
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    timeout=120.0,  # 120 seconds — enough for slow free models
)
CHAT_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
# Embeddings: OpenRouter does NOT provide embeddings, so use OpenAI directly
# (you need an OpenAI key), OR swap for a free local model below.
__embed_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", os.getenv("OPENROUTER_API_KEY")),
    base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

async def chat_completion_stream(messages: list[dict], temperature: float = 0.7):
    """Stream response chunks as they arrive."""
    from openai import AsyncOpenAI
    
    async_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        timeout=120.0
    )
    
    stream = await async_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        stream=True,
        extra_headers={"HTTP-Referer": "http://localhost", "X-OpenRouter-Title": "MCP RAG Agent"}
    )
    
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def get_embedding(text: str) -> list[float]:
    """Return the vector embedding for a piece of text."""
    response = __embed_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding
    """Return the vector embedding for a piece of text."""
    response = _embed_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def chat_completion(messages: list[dict],
                    temperature: float = 0.7,
                    max_tokens: int = 2000) -> str:
    """Call chat completion via OpenRouter."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers={
            "HTTP-Referer": "http://localhost",
            "X-OpenRouter-Title": "MCP RAG Agent",
        },
    )
    return response.choices[0].message.content


def rag_query(question: str, context_docs: list[dict]) -> str:
    """Generate an answer grounded in retrieved context."""
    context = "\n\n".join(
        f"[Source: {d['source']}]\n{d['content']}" for d in context_docs
    )
    messages = [
        {"role": "system",
         "content": "You are a knowledge assistant. Answer using ONLY the "
                    "provided reference material. If insufficient, say so."},
        {"role": "user",
         "content": f"Reference:\n{context}\n\nQuestion: {question}"},
    ]
    return chat_completion(messages, temperature=0.3)