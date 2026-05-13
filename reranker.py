from typing import List
import numpy as np
from app.embedding_provider import BaseEmbeddingProvider

class SemanticReranker:
    def __init__(self, embedding_provider: BaseEmbeddingProvider):
        """
        Initializes the reranker with an embedding provider to compute 
        semantic similarity between queries and candidates.
        """
        self.embedding_provider = embedding_provider

    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        """
        Reranks a list of candidate documents based on their cosine similarity 
        to the input query.
        """
        if not candidates:
            return candidates

        # Query and candidates-ku embeddings edukkirom
        query_vector = np.array(self.embedding_provider.embed([query])[0])
        candidate_texts = [candidate["text"] for candidate in candidates]
        candidate_vectors = np.array(self.embedding_provider.embed(candidate_texts))

        # Vectorized Cosine Similarity calculation
        # Dot product of normalized vectors
        query_norm = np.linalg.norm(query_vector)
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1)

        # Division by zero error-ah avoid panna chinna epsilon value
        query_vector = query_vector / (query_norm + 1e-9)
        candidate_vectors = candidate_vectors / (candidate_norms[:, np.newaxis] + 1e-9)

        # Matrix multiplication to get all scores at once
        scores = np.dot(candidate_vectors, query_vector).tolist()

        # Candidates kooda score-ah attach panni rank pandrom
        ranked = []
        for candidate, score in zip(candidates, scores):
            # Original candidate dict-ah modify pannaama pudhu copy create pandrom
            new_candidate = candidate.copy()
            new_candidate["rerank_score"] = float(score)
            ranked.append(new_candidate)

        # Score-ah base panni descending order-la sort pandrom
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        
        return ranked