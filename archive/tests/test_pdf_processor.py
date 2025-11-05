"""Unit tests for the PDF processor module."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from pdf_processor import QAPair, LAQData, PDFProcessingError, PDFProcessor
from config import Config


class TestQAPair:
    """Tests for the QAPair model."""

    def test_valid_qa_pair(self):
        """Test that valid Q&A pairs are accepted."""
        qa = QAPair(question="What is the budget?", answer="Rs. 50 crores")
        assert qa.question == "What is the budget?"
        assert qa.answer == "Rs. 50 crores"

    def test_empty_question_rejected(self):
        """Test that empty questions are rejected."""
        with pytest.raises(ValidationError):
            QAPair(question="", answer="Some answer")

    def test_empty_answer_rejected(self):
        """Test that empty answers are rejected."""
        with pytest.raises(ValidationError):
            QAPair(question="Some question?", answer="")


class TestLAQData:
    """Tests for the LAQData model."""

    def test_valid_laq_data(self):
        """Test that valid LAQ data is accepted."""
        laq = LAQData(
            pdf_title="Test LAQ",
            laq_type="Starred",
            laq_number="123",
            minister="Test Minister",
            date="2024-01-01",
            qa_pairs=[
                QAPair(question="Q1?", answer="A1"),
                QAPair(question="Q2?", answer="A2"),
            ],
        )
        assert laq.pdf_title == "Test LAQ"
        assert len(laq.qa_pairs) == 2

    def test_empty_qa_pairs_rejected(self):
        """Test that LAQs with no Q&A pairs are rejected."""
        with pytest.raises(ValidationError):
            LAQData(
                pdf_title="Test LAQ",
                laq_type="Starred",
                laq_number="123",
                minister="Test Minister",
                date="2024-01-01",
                qa_pairs=[],
            )

    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        laq = LAQData(
            pdf_title="Test LAQ",
            laq_type="Starred",
            laq_number="123",
            minister="Test Minister",
            date="2024-01-01",
            qa_pairs=[QAPair(question="Q?", answer="A")],
            tabled_by="Test Person",
            attachments=["Annex-1", "Annex-2"],
        )
        assert laq.tabled_by == "Test Person"
        assert len(laq.attachments) == 2


class TestPDFProcessor:
    """Tests for the PDFProcessor class."""

    def test_validate_nonexistent_file(self):
        """Test that validation fails for non-existent files."""
        config = Config()
        processor = PDFProcessor(config)

        with pytest.raises(PDFProcessingError, match="File not found"):
            processor.validate_pdf_file("/path/to/nonexistent.pdf")

    def test_validate_non_pdf_file(self, tmp_path):
        """Test that validation fails for non-PDF files."""
        config = Config()
        processor = PDFProcessor(config)

        # Create a temporary non-PDF file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        with pytest.raises(PDFProcessingError, match="Not a PDF file"):
            processor.validate_pdf_file(str(test_file))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
