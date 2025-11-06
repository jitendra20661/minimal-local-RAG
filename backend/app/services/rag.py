"""RAG (Retrieval-Augmented Generation) operations for LAQ search and chat."""

import json
from typing import List, Dict, Tuple, Optional
import ollama

from app.services.config import Config
from app.services.database import LAQDatabase
from app.services.embeddings import EmbeddingService


class RAGError(Exception):
    """Raised when RAG operations fail."""
    pass


class RAGService:
    """Handles search and chat operations for the LAQ RAG system."""

    def __init__(self, config: Config, database: LAQDatabase, embedding_service: EmbeddingService):
        """Initialize the RAG service.

        Args:
            config: Application configuration
            database: Database instance
            embedding_service: Embedding service instance
        """
        self.config = config
        self.db = database
        self.embeddings = embedding_service

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        apply_threshold: bool = True,
    ) -> List[Dict]:
        """Search for relevant LAQs.

        Args:
            query: Search query text
            top_k: Number of results to return (uses config default if None)
            apply_threshold: Whether to filter by similarity threshold

        Returns:
            List of search result dictionaries with metadata and scores

        Raises:
            RAGError: If search fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_text(query)

            # Search database
            results = self.db.search(query_embedding, n_results=top_k)

            # Filter by relevance if requested
            if apply_threshold:
                ids, distances, metadatas, documents = self.db.filter_by_relevance(
                    results
                )
            else:
                if not results["ids"] or not results["ids"][0]:
                    return []
                ids = results["ids"][0]
                distances = results["distances"][0]
                metadatas = results["metadatas"][0]
                documents = results.get("documents", [[]])[0]

            # Format results
            formatted_results = []
            for i, doc_id in enumerate(ids):
                distance = distances[i]
                similarity = (1 - distance) * 100  # Convert to percentage

                # Determine match quality
                if similarity >= 80:
                    match_quality = "STRONG MATCH"
                    match_color = "🟢"
                elif similarity >= 60:
                    match_quality = "MODERATE MATCH"
                    match_color = "🟡"
                else:
                    match_quality = "WEAK MATCH"
                    match_color = "🔴"

                meta = metadatas[i]
                doc = documents[i] if i < len(documents) else ""

                formatted_results.append(
                    {
                        "id": doc_id,
                        "similarity": round(similarity, 2),
                        "match_quality": match_quality,
                        "match_color": match_color,
                        "metadata": meta,
                        "document": doc,
                    }
                )

            return formatted_results

        except Exception as e:
            raise RAGError(f"Search failed: {e}") from e

    def chat(self, query: str, top_k: Optional[int] = None) -> Tuple[str, List[Dict]]:
        """Generate a conversational response based on retrieved LAQs.

        Args:
            query: User question
            top_k: Number of LAQs to retrieve for context (uses config default if None)

        Returns:
            Tuple of (response_text, source_laqs)

        Raises:
            RAGError: If chat generation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            # Retrieve relevant LAQs
            if top_k is None:
                top_k = self.config.chat_top_k

            results = self.search(query, top_k=top_k, apply_threshold=True)

            if not results:
                return (
                    "I couldn't find any relevant LAQs to answer your question. "
                    "Please try rephrasing your query or upload more documents.",
                    [],
                )

            # Build context from retrieved LAQs
            context = self._build_context(results)

            # Generate response with improved prompt
            prompt = self._build_chat_prompt(context, query)

            response = ollama.generate(
                model=self.config.llm_model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": self.config.llm_temperature,
                    "top_p": self.config.llm_top_p,
                },
            )

            return response["response"], results

        except Exception as e:
            raise RAGError(f"Chat generation failed: {e}") from e

    def _build_context(self, search_results: List[Dict]) -> str:
        """Build formatted context string from search results.

        Args:
            search_results: List of search result dictionaries

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, result in enumerate(search_results, 1):
            meta = result["metadata"]

            # Parse attachments
            try:
                attachments = json.loads(meta.get("attachments", "[]"))
                attachments_text = (
                    f"\nAttachments: {', '.join(attachments)}" if attachments else ""
                )
            except:
                attachments_text = ""

            part = f"""
LAQ #{meta.get('laq_num', 'N/A')} ({meta.get('type', 'N/A')}) - {meta.get('date', 'N/A')}
Minister: {meta.get('minister', 'N/A')}
Tabled by: {meta.get('tabled_by', 'N/A')}
Question: {meta.get('question', 'N/A')}
Answer: {meta.get('answer', 'N/A')}{attachments_text}
"""
            context_parts.append(part.strip())

        return "\n\n---\n\n".join(context_parts)

    def _build_chat_prompt(self, context: str, query: str) -> str:
        """Build an improved prompt for chat generation.

        Args:
            context: Formatted context from retrieved LAQs
            query: User question

        Returns:
            Complete prompt for LLM
        """
        prompt = f"""You are an expert assistant for Legislative Assembly Questions (LAQs). Your role is to provide accurate, factual answers based on the LAQ database.

Below are the most relevant LAQs from the database:

{context}

---

INSTRUCTIONS:
1. Answer the user's question based ONLY on the information provided in the LAQs above
2. If the LAQs don't contain sufficient information to answer the question, explicitly state what information is missing
3. Always cite specific LAQ numbers when referencing facts (e.g., "According to LAQ #324...")
4. Be precise, concise, and factual - do not add interpretations or speculation
5. If multiple LAQs contain relevant information, synthesize them coherently
6. If attachments are mentioned (like Annexure-I), reference them in your answer
7. Maintain a professional and helpful tone

USER QUESTION: {query}

ANSWER:"""

        return prompt

    def get_match_quality_stats(self, results: List[Dict]) -> Dict[str, int]:
        """Get statistics about match quality distribution.

        Args:
            results: List of search results

        Returns:
            Dictionary with counts for each quality level
        """
        stats = {"strong": 0, "moderate": 0, "weak": 0}

        for result in results:
            similarity = result["similarity"]
            if similarity >= 80:
                stats["strong"] += 1
            elif similarity >= 60:
                stats["moderate"] += 1
            else:
                stats["weak"] += 1

        return stats
