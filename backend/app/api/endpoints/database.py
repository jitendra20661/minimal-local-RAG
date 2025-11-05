"""
Database API endpoint for database information and management.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import DatabaseInfo
from app.services.database import LAQDatabase, DatabaseError
from app.services.config import Config

router = APIRouter()


@router.get("/info", response_model=DatabaseInfo)
async def get_database_info():
    """
    Get database information and statistics.

    - Collection name
    - Total documents
    - Database path
    - Configuration settings
    """

    try:
        config = Config()
        db = LAQDatabase(config)

        # Get collection count
        collection = db.client.get_collection(config.collection_name)
        count = collection.count()

        return DatabaseInfo(
            collection_name=config.collection_name,
            total_documents=count,
            database_path=str(config.db_path),
            similarity_metric="cosine",
            embedding_model=config.embedding_model,
            llm_model=config.llm_model
        )

    except DatabaseError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
