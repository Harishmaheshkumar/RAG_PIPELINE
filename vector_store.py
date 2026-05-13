from typing import List, Optional
from uuid import uuid4
import chromadb
from app.embedding_provider import BaseEmbeddingProvider

class ChromaVectorStore:
    def __init__(self, embedding_provider: BaseEmbeddingProvider, persist_directory: str):
        self.embedding_provider = embedding_provider
        self.persist_directory = persist_directory
        
        # ChromaDB Persistent Client initialization
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Embedding function-ah custom-ah create pandrom
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            embedding_function=self.embedding_provider.chroma_embedding_function(),
            metadata={"hnsw:space": "cosine"} # Similarity metric-ah explicitly define pandrom
        )

    def add_documents(self, documents: List[dict]) -> List[str]:
        """Adds text documents to the vector store with unique IDs and metadata."""
        if not documents:
            return []

        ids = [doc.get("id") or str(uuid4()) for doc in documents]
        texts = [doc["text"] for doc in documents]
        # Metadata none-ah irundha empty dict-ah handle pandrom
        metadatas = [doc.get("metadata") or {} for doc in documents]
        
        try:
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
            return ids
        except Exception as e:
            print(f"Error adding documents to Chroma: {e}")
            raise

    def query(self, query_text: str, n_results: int = 10) -> List[dict]:
        """Queries the vector store for the most relevant documents."""
        try:
            result = self.collection.query(query_texts=[query_text], n_results=n_results)
            
            docs = []
            # Results empty-ah irukka-nu check pandrom
            if not result or not result["ids"] or len(result["ids"][0]) == 0:
                return docs

            for idx in range(len(result["ids"][0])):
                docs.append(
                    {
                        "id": result["ids"][0][idx],
                        "text": result["documents"][0][idx],
                        "metadata": result["metadatas"][0][idx] or {},
                        # Chroma distances usually lower is better, namma score-ah handle pandrom
                        "score": float(result["distances"][0][idx]) if result.get("distances") else 0.0,
                    }
                )
            return docs
        except Exception as e:
            print(f"Error querying Chroma: {e}")
            return []

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Retrieves a single document by its ID."""
        try:
            result = self.collection.get(ids=[doc_id])
            if not result or not result["ids"]:
                return None
                
            return {
                "id": result["ids"][0],
                "text": result["documents"][0][idx] if isinstance(result["documents"], list) else result["documents"],
                "metadata": result["metadatas"][0] if result["metadatas"] else {},
            }
        except Exception as e:
            print(f"Error fetching document {doc_id}: {e}")
            return None