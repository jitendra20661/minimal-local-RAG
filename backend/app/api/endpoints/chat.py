"""
Chat API endpoint for conversational LAQ queries.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatQuery, ChatResponse, SearchResult
from app.services.rag import RAGService, RAGError
from app.services.config import Config

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat_with_laqs(query: ChatQuery):
    """
    Chat with LAQ knowledge base.

    - Retrieves relevant LAQs
    - Generates contextual answer using LLM
    - Returns answer with source citations
    """

    try:
        # Initialize RAG service
        config = Config()
        rag_service = RAGService(config)

        # Perform chat
        response = rag_service.chat(
            question=query.question,
            top_k=query.top_k
        )

        # Convert sources to response model
        sources = [
            SearchResult(
                question=source['metadata']['question'],
                answer=source['metadata']['answer'],
                metadata=source['metadata'],
                similarity_score=source['score']
            )
            for source in response['sources']
        ]

        return ChatResponse(
            question=query.question,
            answer=response['answer'],
            sources=sources
        )

    except RAGError as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
