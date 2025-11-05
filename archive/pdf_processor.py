"""PDF processing and LAQ data extraction using Docling and Mistral LLM."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import ollama
from docling.document_converter import DocumentConverter
from pydantic import BaseModel, Field, ValidationError

from config import Config


class QAPair(BaseModel):
    """Model for a question-answer pair."""

    question: str = Field(min_length=1, description="The question text")
    answer: str = Field(min_length=1, description="The answer text")


class LAQData(BaseModel):
    """Model for structured LAQ data."""

    pdf_title: str = Field(description="Title of the LAQ document")
    laq_type: str = Field(description="Type of LAQ (e.g., Starred, Unstarred)")
    laq_number: str = Field(description="LAQ identification number")
    minister: str = Field(description="Name of the minister")
    date: str = Field(description="Date of the LAQ")
    qa_pairs: List[QAPair] = Field(
        min_items=1, description="List of question-answer pairs"
    )
    tabled_by: Optional[str] = Field(None, description="Person who tabled the question")
    attachments: List[str] = Field(default_factory=list, description="List of attachments")


class PDFProcessingError(Exception):
    """Raised when PDF processing fails."""
    pass


class PDFProcessor:
    """Handles PDF to structured LAQ data conversion."""

    def __init__(self, config: Config):
        """Initialize the PDF processor.

        Args:
            config: Application configuration
        """
        self.config = config
        self.converter = DocumentConverter()

    def validate_pdf_file(self, pdf_path: str) -> Path:
        """Validate that the file exists and is a PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Validated Path object

        Raises:
            PDFProcessingError: If validation fails
        """
        path = Path(pdf_path)

        if not path.exists():
            raise PDFProcessingError(f"File not found: {pdf_path}")

        if not path.is_file():
            raise PDFProcessingError(f"Not a file: {pdf_path}")

        if path.suffix.lower() != ".pdf":
            raise PDFProcessingError(
                f"Not a PDF file: {pdf_path}\n"
                f"Expected .pdf extension, got {path.suffix}"
            )

        # Check file size (warn if > 10MB)
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 10:
            print(f"⚠️ Large file detected ({size_mb:.1f} MB). Processing may take time.")

        return path

    def extract_markdown_from_pdf(self, pdf_path: Path) -> str:
        """Convert PDF to markdown using Docling.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Markdown content

        Raises:
            PDFProcessingError: If conversion fails
        """
        try:
            print(f"🔄 Converting PDF to markdown...")
            doc = self.converter.convert(str(pdf_path))
            markdown_data = doc.document.export_to_markdown()
            print(f"✅ Conversion successful ({len(markdown_data)} characters)")
            return markdown_data
        except Exception as e:
            raise PDFProcessingError(
                f"Failed to convert PDF to markdown: {e}"
            ) from e

    def structure_laqs_with_mistral(self, markdown_data: str, pdf_path: Path) -> LAQData:
        """Use Mistral LLM to extract structured LAQ data from markdown.

        Args:
            markdown_data: Markdown content from PDF
            pdf_path: Path to the original PDF file

        Returns:
            Validated LAQData object

        Raises:
            PDFProcessingError: If LLM processing or validation fails
        """
        try:
            print("🤖 Processing with Mistral LLM...")

            prompt = f"""
You are a structured data extraction assistant. Extract Legislative Assembly Question (LAQ) details from the following text.

The text comes from an official LAQ PDF and may include multi-line tables, line breaks, and subparts (a), (b), (c), etc.

Your goal is to output **well-structured JSON** where:
- Each sub-question (a), (b), (c) becomes a **separate Q&A pair** in the "qa_pairs" list.
- Questions and answers are **complete**, not truncated.
- Original wording is **preserved exactly** — do not paraphrase or summarize.
- Do not merge subparts into a single question.

---

### REQUIRED OUTPUT FORMAT

{{
  "pdf_title": "TENDER ISSUED FOR LEASING OF JETTY SPACE",
  "laq_type": "Starred",
  "laq_number": "010C",
  "minister": "Shri. Aleixo Sequeira, Minister for Captain of Ports Department",
  "tabled_by": "Shri Digambar Kamat",
  "date": "08-08-2025",
  "qa_pairs": [
    {{
      "question": "(a) the details with the total number of jetty spots available in the river Mandovi for use by Casino and cruises vessels including location, area of use in sq.mt of all the individual jetty spots with details of all vessels that are using each particular jetty spot and the purpose of usage;",
      "answer": "Sir, there are total 12 number of jetty spots in river Mandovi for use by Casino and cruises vessels. The details are enclosed at Annexure - I."
    }},
    {{
      "question": "(b) the details of all tender issued for leasing jetty space in river Mandovi from the year 2020 till date including tender number, financial bid, copy of lease agreement, amounts received year-wise from inception of tender;",
      "answer": "Santa Monica Jetty (Tourism Department)\\n1. Tender No. GTDC/JETTY/2019-20/3185\\n2. Financial Bid: Rs. 1.23 Cr. Plus taxes\\n3. Copy of lease agreement enclosed at Annexure - II\\n4. Year Amount Received\\n16/07/2023 to 15/07/2024: 1,23,00,000 + GST 22,14,000\\n16/07/2024 to 15/07/2025: 1,23,00,000 + GST 22,14,000"
    }},
    {{
      "question": "(c) the details of the last tender floated by the Government/COP/RND Department for leasing the River Navigation jetty opposite the Old Secretariat including details of all file noting with copy of lease agreement, total amount received by the Government from lease holders from its inception year-wise?",
      "answer": "Nil"
    }}
  ],
  "attachments": ["Annexure - I", "Annexure - II"]
}}

---

### RULES
1. Detect and reconstruct full text of each sub-question and its matching answer.
2. Treat text like "(a) …", "(b) …", "(c) …" as boundaries for new Q&A pairs.
3. Combine lines until a new sub-question or section begins.
4. Keep punctuation and formatting (like "\\n" for line breaks) intact.
5. Output **only valid JSON**. Do not include explanations or extra commentary.

Now extract the structured data in this format from the following text:

{markdown_data[:self.config.markdown_chunk_size]}
"""

            response = ollama.generate(
                model=self.config.llm_model,
                prompt=prompt,
                stream=False,
                options={"timeout": self.config.ollama_timeout},
            )
            response_text = response["response"].strip()

            # Try to parse JSON
            try:
                laq_dict = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from response
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    laq_dict = json.loads(json_match.group())
                else:
                    raise PDFProcessingError(
                        f"Could not parse Mistral response as JSON.\n"
                        f"First 200 chars: {response_text[:200]}"
                    )

            # Validate with Pydantic
            try:
                laq_data = LAQData(**laq_dict)
                print(f"✅ Successfully extracted {len(laq_data.qa_pairs)} Q&A pairs")
                return laq_data
            except ValidationError as e:
                raise PDFProcessingError(
                    f"Invalid LAQ data structure:\n{e}"
                ) from e

        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise
            raise PDFProcessingError(f"Mistral processing error: {e}") from e

    def process_pdf(self, pdf_path: str) -> LAQData:
        """Complete pipeline: validate -> extract markdown -> structure with LLM.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Validated LAQData object

        Raises:
            PDFProcessingError: If any step fails
        """
        # Step 1: Validate
        validated_path = self.validate_pdf_file(pdf_path)

        # Step 2: Extract markdown
        markdown_data = self.extract_markdown_from_pdf(validated_path)

        # Step 3: Structure with LLM
        laq_data = self.structure_laqs_with_mistral(markdown_data, validated_path)

        return laq_data
