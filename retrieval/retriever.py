# retrieval/retriever.py
# This is our smart librarian that finds relevant code docs

import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "code_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class HybridRetriever:
    def __init__(self):
        print("🔍 Setting up retriever...")
        
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.bm25 = None
        self.bm25_corpus = []
        
        print("✅ Retriever ready!")
    
    def index_documents(self, documents: list):
        print(f"📚 Indexing {len(documents)} documents...")
        
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
        
        texts = [doc["text"] for doc in documents]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        points = [
            PointStruct(
                id=i,
                vector=emb.tolist(),
                payload={"text": doc["text"], "metadata": doc.get("metadata", {})}
            )
            for i, (doc, emb) in enumerate(zip(documents, embeddings))
        ]
        
        self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        
        self.bm25_corpus = texts
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        
        print(f"✅ Indexed {len(documents)} documents!")
    
    def retrieve(self, query: str, top_k: int = 5) -> list:
        query_embedding = self.encoder.encode([query])[0]
        dense_results = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        dense_texts = {r.payload["text"]: r.score for r in dense_results}
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_k]
        sparse_texts = {
            self.bm25_corpus[i]: bm25_scores[i]
            for i in top_bm25_indices
        }
        
        all_texts = set(dense_texts.keys()) | set(sparse_texts.keys())
        combined_scores = {}
        for text in all_texts:
            dense_score = dense_texts.get(text, 0)
            sparse_score = sparse_texts.get(text, 0)
            combined_scores[text] = (0.6 * dense_score) + (0.4 * sparse_score)
        
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [text for text, _ in ranked[:top_k]]
    
    def format_context(self, retrieved_docs: list) -> str:
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Retrieved Context {i}]:\n{doc}")
        return "\n\n".join(context_parts)