"""Embedding generation using Ollama for the LAQ RAG system."""

from typing import List
import ollama

from app.services.config import Config


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class OllamaConnectionError(EmbeddingError):
    """Raised when cannot connect to Ollama."""
    pass


class OllamaModelNotFoundError(EmbeddingError):
    """Raised when the embedding model is not found."""
    pass


class EmbeddingService:
    """Handles embedding generation using Ollama."""

    def __init__(self, config: Config):
        """Initialize the embedding service.

        Args:
            config: Application configuration
        """
        self.config = config
        self._verify_ollama_connection()

    def _verify_ollama_connection(self) -> None:
        """Verify that Ollama is accessible and the model is available.

        Raises:
            OllamaConnectionError: If cannot connect to Ollama
            OllamaModelNotFoundError: If embedding model not found
        """
        try:
            # Try to list models to verify connection
            ollama.list()
        except Exception as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.config.ollama_host}.\n"
                f"Please ensure:\n"
                f"  1. Ollama is installed: https://ollama.ai\n"
                f"  2. Ollama is running: 'ollama serve'\n"
                f"Error: {e}"
            ) from e

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            ValueError: If text is empty
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            response = ollama.embed(model=self.config.embedding_model, input=text)
            return response["embeddings"][0]
        except KeyError as e:
            raise OllamaModelNotFoundError(
                f"Model '{self.config.embedding_model}' not found.\n"
                f"Run: ollama pull {self.config.embedding_model}"
            ) from e
        except ConnectionError as e:
            raise OllamaConnectionError(
                f"Lost connection to Ollama.\n"
                f"Ensure Ollama is running: 'ollama serve'"
            ) from e
        except Exception as e:
            raise EmbeddingError(f"Unexpected embedding error: {e}") from e

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If texts list is empty
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
            except EmbeddingError as e:
                print(f"⚠️ Failed to embed text {i+1}/{len(texts)}: {e}")
                # Return empty embedding as placeholder
                embeddings.append([])

        return embeddings

    def embed_qa_pairs(self, qa_pairs: List[dict]) -> List[List[float]]:
        """Generate embeddings for Q&A pairs.

        Args:
            qa_pairs: List of dictionaries with 'question' and 'answer' keys

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If qa_pairs list is empty
        """
        if not qa_pairs:
            raise ValueError("Q&A pairs list cannot be empty")

        texts = [
            f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}"
            for qa in qa_pairs
        ]
        return self.embed_batch(texts)
