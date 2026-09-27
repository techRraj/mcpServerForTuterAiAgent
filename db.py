"""PostgreSQL + pgvector database operations."""
import os
import uuid
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


# ---------- Init ----------

def init_db():
    """Create pgvector extension and all required tables + indexes."""
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Documents
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                source VARCHAR(255),
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Embeddings (2048 dims because nemotron-3-embed-1b returns 2048)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                embedding halfvec(2048),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # HNSW index for fast similarity search
        conn.execute("""
            CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
            ON embeddings
            USING hnsw (embedding halfvec_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)

        # Conversations (memory) table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                role VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS conv_session_idx
            ON conversations(session_id)
        """)

    print("[OK] Database initialized")


# ---------- Chunking ----------

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Sentence-aware chunking with overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        boundary = -1
        for i in range(end, max(start + chunk_size // 2, end - 200), -1):
            if text[i] in '.!?\n':
                boundary = i + 1
                break

        actual_end = boundary if boundary > start else end
        chunks.append(text[start:actual_end].strip())

        start = actual_end - chunk_overlap
        if start <= 0:
            start = actual_end

    return [c for c in chunks if c]


# ---------- Insert ----------

def insert_document(content: str, source: str = "", metadata: dict = None) -> int:
    with psycopg.connect(DATABASE_URL) as conn:
        cur = conn.execute(
            "INSERT INTO documents (content, source, metadata) VALUES (%s, %s, %s) RETURNING id",
            (content, source, Jsonb(metadata or {}))
        )
        doc_id = cur.fetchone()[0]
        conn.commit()
        return doc_id


def insert_embedding(document_id: int, embedding: list[float]):
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        conn.execute(
            "INSERT INTO embeddings (document_id, embedding) VALUES (%s, %s)",
            (document_id, embedding)
        )
        conn.commit()


# ---------- Search ----------

def search_similar(query_embedding: list[float], top_k: int = 5,
                   min_score: float = 0.0, source_filter: str = None) -> list[dict]:
    """Vector search with optional source filter."""
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        if source_filter:
            cur = conn.execute("""
                SELECT d.id, d.content, d.source, d.metadata,
                       1 - (e.embedding <=> %s::halfvec) AS similarity
                FROM embeddings e
                JOIN documents d ON d.id = e.document_id
                WHERE 1 - (e.embedding <=> %s::halfvec) >= %s
                  AND d.source = %s
                ORDER BY e.embedding <=> %s::halfvec
                LIMIT %s
            """, (query_embedding, query_embedding, min_score, source_filter, query_embedding, top_k))
        else:
            cur = conn.execute("""
                SELECT d.id, d.content, d.source, d.metadata,
                       1 - (e.embedding <=> %s::halfvec) AS similarity
                FROM embeddings e
                JOIN documents d ON d.id = e.document_id
                WHERE 1 - (e.embedding <=> %s::halfvec) >= %s
                ORDER BY e.embedding <=> %s::halfvec
                LIMIT %s
            """, (query_embedding, query_embedding, min_score, query_embedding, top_k))

        return [
            {"id": r[0], "content": r[1], "source": r[2],
             "metadata": r[3], "similarity": float(r[4])}
            for r in cur.fetchall()
        ]


# ---------- Stats ----------

def get_stats() -> dict:
    with psycopg.connect(DATABASE_URL) as conn:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {"document_count": doc_count, "embedding_count": emb_count}


# ---------- Conversation memory ----------

def save_message(session_id: str, role: str, content: str):
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, role, content) VALUES (%s, %s, %s)",
            (session_id, role, content)
        )
        conn.commit()


def get_history(session_id: str, limit: int = 20) -> list[dict]:
    with psycopg.connect(DATABASE_URL) as conn:
        cur = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id = %s ORDER BY created_at LIMIT %s",
            (session_id, limit)
        )
        return [{"role": r[0], "content": r[1]} for r in cur.fetchall()]


def create_session(session_id: str = None) -> str:
    return session_id or str(uuid.uuid4())