"""PostgreSQL + pgvector database operations."""
import os
import psycopg
from pgvector.psycopg import register_vector
from pgvector import HalfVector
from dotenv import load_dotenv
import json
from psycopg.types.json import Jsonb
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def init_db():
    """Create pgvector extension and required tables."""
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                source VARCHAR(255),
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Embedding dimension must match your embedding model
        # text-embedding-3-small = 1536 dims
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
               embedding halfvec(2048),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # HNSW index — works on empty tables, unlike IVFFlat
        conn.execute("""
            CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
            ON embeddings
            USING hnsw (embedding halfvec_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)

    print("[OK] Database initialized")


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


def search_similar(query_embedding: list[float], top_k: int = 5, min_score: float = 0.0) -> list[dict]:
    # Convert the plain list to a HalfVector object
    query_vec = HalfVector(query_embedding)
    
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        cur = conn.execute("""
            SELECT d.id, d.content, d.source, d.metadata,
                   1 - (e.embedding <=> %s) AS similarity
            FROM embeddings e
            JOIN documents d ON d.id = e.document_id
            WHERE 1 - (e.embedding <=> %s) >= %s
            ORDER BY e.embedding <=> %s
            LIMIT %s
        """, (query_vec, query_vec, min_score, query_vec, top_k))

        return [
            {"id": r[0], "content": r[1], "source": r[2],
             "metadata": r[3], "similarity": float(r[4])}
            for r in cur.fetchall()
        ]


def get_stats() -> dict:
    with psycopg.connect(DATABASE_URL) as conn:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {"document_count": doc_count, "embedding_count": emb_count}