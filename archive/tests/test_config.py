"""Unit tests for the configuration module."""

import os
import pytest
from pathlib import Path

from config import Config


class TestConfig:
    """Tests for the Config class."""

    def test_default_config(self):
        """Test that default configuration is valid."""
        config = Config()

        assert config.db_path == Path("./laq_db")
        assert config.collection_name == "laqs"
        assert config.llm_model == "mistral"
        assert config.embedding_model == "nomic-embed-text"
        assert config.search_top_k == 5
        assert config.chat_top_k == 3
        assert 0 <= config.similarity_threshold <= 1

    def test_similarity_threshold_validation(self):
        """Test that invalid similarity thresholds raise errors."""
        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            Config(similarity_threshold=1.5)

        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            Config(similarity_threshold=-0.1)

    def test_top_k_validation(self):
        """Test that invalid top_k values raise errors."""
        with pytest.raises(ValueError, match="search_top_k must be >= 1"):
            Config(search_top_k=0)

        with pytest.raises(ValueError, match="chat_top_k must be >= 1"):
            Config(chat_top_k=0)

    def test_temperature_validation(self):
        """Test that invalid temperature values raise errors."""
        with pytest.raises(ValueError, match="llm_temperature must be between 0 and 2"):
            Config(llm_temperature=-0.1)

        with pytest.raises(ValueError, match="llm_temperature must be between 0 and 2"):
            Config(llm_temperature=3.0)

    def test_db_path_creation(self):
        """Test that database directory is created."""
        test_path = Path("./test_db_temp")
        config = Config(db_path=test_path)

        assert test_path.exists()
        assert test_path.is_dir()

        # Cleanup
        test_path.rmdir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
