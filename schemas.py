from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    id: Optional[str] = Field(None, description="Optional document ID")
    text: str = Field(..., description="Document text content")
    metadata: Optional[dict] = Field(default_factory=dict, description="Metadata for the document")


class IngestRequest(BaseModel):
    documents: List[DocumentCreate]


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class DocumentFragment(BaseModel):
    id: str
    text: str
    metadata: dict
    # Reranker-la 'rerank_score' use panradhala inga optional-ah vechikalam
    score: Optional[float] = 0.0
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    hallucination_score: float
    consistency: str
    sources: List[DocumentFragment]
    reranked_sources: List[DocumentFragment]
    evaluation_rationale: str