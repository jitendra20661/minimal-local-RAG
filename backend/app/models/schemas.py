"""
Pydantic models for API request/response schemas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# Request/Response Models
class QAPairResponse(BaseModel):
    """Q&A pair response model."""
    question: str
    answer: str


class LAQDataResponse(BaseModel):
    """LAQ data response model."""
    pdf_title: str
    laq_type: str
    laq_number: str
    minister: str
    date: str
    qa_pairs: List[QAPairResponse]
    tabled_by: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)


class SearchQuery(BaseModel):
    """Search query request model."""
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of results to return")
    threshold: Optional[float] = Field(0.6, ge=0.0, le=1.0, description="Similarity threshold")


class SearchResult(BaseModel):
    """Single search result."""
    question: str
    answer: str
    metadata: dict
    similarity_score: float


class SearchResponse(BaseModel):
    """Search results response."""
    query: str
    results: List[SearchResult]
    total_results: int


class ChatQuery(BaseModel):
    """Chat query request model."""
    question: str = Field(..., min_length=1, description="User question")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of context LAQs")


class ChatResponse(BaseModel):
    """Chat response model."""
    question: str
    answer: str
    sources: List[SearchResult]


class UploadResponse(BaseModel):
    """PDF upload response model."""
    success: bool
    message: str
    pdf_name: str
    qa_pairs_extracted: int
    laq_data: Optional[LAQDataResponse] = None


class DatabaseInfo(BaseModel):
    """Database information response."""
    collection_name: str
    total_documents: int
    database_path: str
    similarity_metric: str
    embedding_model: str
    llm_model: str


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    ollama_running: bool
    database: str
