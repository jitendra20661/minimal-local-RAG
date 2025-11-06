"""ChromaDB database operations for the LAQ RAG system."""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.api.models.Collection import Collection

from app.services.config import Config


class DatabaseError(Exception):
    """Raised when database operations fail."""
    pass


class LAQDatabase:
    """Manages ChromaDB operations for LAQ storage and retrieval."""

    def __init__(self, config: Config):
        """Initialize the database connection.

        Args:
            config: Application configuration

        Raises:
            DatabaseError: If database initialization fails
        """
        self.config = config
        try:
            self.client = chromadb.PersistentClient(path=str(config.db_path))
            self.collection = self.client.get_or_create_collection(
                name=config.collection_name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise DatabaseError(f"Failed to initialize database: {e}") from e

    def store_qa_pairs(
        self,
        laq_data: Dict,
        pdf_name: str,
        embeddings_list: List[List[float]],
    ) -> int:
        """Store Q&A pairs in ChromaDB with batch insertion.

        Args:
            laq_data: Structured LAQ data dictionary
            pdf_name: Name of the source PDF file
            embeddings_list: Pre-generated embeddings for each Q&A pair

        Returns:
            Number of successfully stored Q&A pairs

        Raises:
            DatabaseError: If storage operation fails
        """
        qa_pairs = laq_data.get("qa_pairs", [])
        if not qa_pairs:
            return 0

        if len(embeddings_list) != len(qa_pairs):
            raise ValueError(
                f"Mismatch: {len(embeddings_list)} embeddings for {len(qa_pairs)} Q&A pairs"
            )

        # Prepare batch data
        ids, embeddings, metadatas, documents = [], [], [], []

        for idx, (qa, embedding) in enumerate(zip(qa_pairs, embeddings_list), 1):
            try:
                laq_num = laq_data.get("laq_number", "unknown")
                doc_id = f"{Path(pdf_name).stem}_{laq_num}_qa{idx}"

                # Check for duplicate IDs
                if self._id_exists(doc_id):
                    print(f"⚠️ Skipping duplicate ID: {doc_id}")
                    continue

                question = qa.get("question", "")
                answer = qa.get("answer", "")
                text = f"Q: {question}\nA: {answer}"

                metadata = {
                    "pdf": pdf_name,
                    "pdf_title": laq_data.get("pdf_title", "N/A"),
                    "laq_num": str(laq_num),
                    "qa_pair_num": str(idx),
                    "type": laq_data.get("laq_type", "N/A"),
                    "question": question[: self.config.metadata_max_length],
                    "answer": answer[: self.config.metadata_max_length],
                    "minister": laq_data.get("minister", "N/A"),
                    "date": laq_data.get("date", "N/A"),
                    "attachments": json.dumps(laq_data.get("attachments", [])),
                }

                ids.append(doc_id)
                embeddings.append(embedding)
                metadatas.append(metadata)
                documents.append(text)

            except Exception as e:
                print(f"⚠️ Error preparing Q&A pair {idx}: {e}")
                continue

        # Batch insert
        if ids:
            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                return len(ids)
            except Exception as e:
                raise DatabaseError(f"Batch insert failed: {e}") from e

        return 0

    def search(
        self, query_embedding: List[float], n_results: Optional[int] = None
    ) -> Dict:
        """Search for similar Q&A pairs.

        Args:
            query_embedding: Embedding vector for the query
            n_results: Number of results to return (uses config default if None)

        Returns:
            Dictionary with search results including IDs, distances, and metadata

        Raises:
            DatabaseError: If search operation fails
        """
        if n_results is None:
            n_results = self.config.search_top_k

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["distances", "metadatas", "documents"],
            )
            return results
        except Exception as e:
            raise DatabaseError(f"Search failed: {e}") from e

    def filter_by_relevance(
        self, results: Dict, threshold: Optional[float] = None
    ) -> Tuple[List, List, List, List]:
        """Filter search results by similarity threshold.

        Args:
            results: Raw search results from ChromaDB
            threshold: Similarity threshold (0-1), uses config default if None

        Returns:
            Tuple of (filtered_ids, filtered_distances, filtered_metadatas, filtered_documents)
        """
        if threshold is None:
            threshold = self.config.similarity_threshold

        if not results["ids"] or not results["ids"][0]:
            return [], [], [], []

        filtered_ids = []
        filtered_distances = []
        filtered_metadatas = []
        filtered_documents = []

        for i, (doc_id, dist) in enumerate(
            zip(results["ids"][0], results["distances"][0])
        ):
            # Convert distance to similarity score (0-1)
            similarity = 1 - dist

            if similarity >= threshold:
                filtered_ids.append(doc_id)
                filtered_distances.append(dist)
                filtered_metadatas.append(results["metadatas"][0][i])
                if "documents" in results and results["documents"]:
                    filtered_documents.append(results["documents"][0][i])

        return (
            filtered_ids,
            filtered_distances,
            filtered_metadatas,
            filtered_documents,
        )

    def clear(self) -> None:
        """Clear all data from the database.

        Raises:
            DatabaseError: If clear operation fails
        """
        try:
            self.client.delete_collection(self.config.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise DatabaseError(f"Failed to clear database: {e}") from e

    def get_count(self) -> int:
        """Get the total number of documents in the collection.

        Returns:
            Number of documents
        """
        try:
            return self.collection.count()
        except Exception as e:
            print(f"⚠️ Failed to get count: {e}")
            return 0

    def _id_exists(self, doc_id: str) -> bool:
        """Check if a document ID already exists in the collection.

        Args:
            doc_id: Document ID to check

        Returns:
            True if ID exists, False otherwise
        """
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result["ids"]) > 0
        except:
            return False
