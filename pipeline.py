from typing import List
from app.config import settings
from app.embedding_provider import GeminiEmbeddingProvider, SentenceTransformerEmbeddingProvider
from app.evaluator import ResponseEvaluator
from app.mlflow_logger import initialize_mlflow, log_evaluation
from app.reranker import SemanticReranker
from app.schemas import DocumentFragment
from app.vector_store import ChromaVectorStore

# Latest Google GenAI library import
try:
    from google import genai
except ImportError:
    genai = None


class RAGPipeline:
    def __init__(self):
        # 1. Variable name fix: settings.GEMINI_API_KEY -> settings.GOOGLE_API_KEY
        api_key = settings.GOOGLE_API_KEY
        
        self.embedding_provider = self._create_embedding_provider()
        self.vector_store = ChromaVectorStore(
            embedding_provider=self.embedding_provider,
            persist_directory=str(settings.CHROMA_PERSIST_DIR),
        )
        self.reranker = SemanticReranker(self.embedding_provider)
        
        # Evaluator-ku correct API key anuppuradhu
        self.evaluator = ResponseEvaluator(api_key=api_key, model=settings.GEMINI_MODEL)
        
        initialize_mlflow()

        if genai is None:
            raise RuntimeError("google-genai client is required for generation. Run 'pip install google-genai'")
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing in environment variables.")

        # Latest Client initialization
        self.client = genai.Client(api_key=api_key)
        self.model_id = settings.GEMINI_MODEL

    def _create_embedding_provider(self):
        # Settings-la GOOGLE_API_KEY irukundhanu check pandrom
        if settings.GOOGLE_API_KEY:
            return GeminiEmbeddingProvider(
                api_key=settings.GOOGLE_API_KEY, 
                model_name=settings.GEMINI_EMBEDDING_MODEL
            )
        return SentenceTransformerEmbeddingProvider()

    def ingest_documents(self, documents: List[dict]) -> List[str]:
        return self.vector_store.add_documents(documents)

    def retrieve(self, query: str, top_k: int) -> List[dict]:
        documents = self.vector_store.query(query_text=query, n_results=top_k)
        return documents

    def generate_answer(self, query: str, sources: List[dict]) -> str:
        snippets = "\n\n".join(
            [f"[{idx + 1}] {item['text']}" for idx, item in enumerate(sources)]
        )
        prompt = (
            "You are a knowledgeable assistant. Use only the provided sources to answer the query. "
            "If the information is not contained in the sources, say you cannot answer.\n\n"
            f"Sources:\n{snippets}\n\n"
            f"Query: {query}\n\nAnswer:"
        )
        
        # Latest google-genai generation syntax
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config={"temperature": settings.RESPONSE_TEMPERATURE}
            )
            return response.text
        except Exception as e:
            return f"Error in generation: {str(e)}"

    def run(self, query: str, top_k: int = None) -> dict:
        top_k = top_k or settings.MAX_RETRIEVAL_RESULTS
        retrieved = self.retrieve(query=query, top_k=top_k)
        
        # Reranking logic
        reranked = self.reranker.rerank(query=query, candidates=retrieved[: settings.RERANK_TOP_K])
        
        # Answer generation
        answer = self.generate_answer(query=query, sources=reranked)
        
        # Evaluation for hallucinations
        evaluation = self.evaluator.assess(query=query, answer=answer, sources=reranked)

        metrics = {
            "retrieved_count": len(retrieved),
            "reranked_count": len(reranked),
            "hallucination_score": evaluation["hallucination_score"],
        }
        log_evaluation(run_name="rag-response", query=query, metrics=metrics)

        return {
            "query": query,
            "answer": answer,
            "hallucination_score": evaluation["hallucination_score"],
            "consistency": evaluation["consistency"],
            "sources": [DocumentFragment(**item) for item in retrieved],
            "reranked_sources": [DocumentFragment(**item) for item in reranked],
            "evaluation_rationale": evaluation["rationale"],
        }