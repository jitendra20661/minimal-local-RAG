# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A minimal, privacy-friendly Retrieval-Augmented Generation (RAG) system that runs entirely locally without external API calls. The system processes Legislative Assembly Question (LAQ) PDFs, extracts Q&A pairs using LLMs, stores them in a vector database, and enables semantic search and chat interactions.

**Tech Stack:**
- **LLM:** Mistral (via Ollama)
- **Embeddings:** nomic-embed-text (via Ollama)
- **Vector DB:** ChromaDB with cosine similarity
- **Document Conversion:** Docling (PDF to Markdown)
- **Validation:** Pydantic for data schema validation
- **Language:** Python 3.10+

## Running the Application

### Prerequisites
Ollama must be installed and running with required models:
```bash
# Start Ollama and pull required models
ollama pull mistral
ollama pull nomic-embed-text
```

### Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Configure environment variables
cp .env.example .env
# Edit .env to customize settings

# Run the application
python main.py
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Sample PDFs
Test PDFs are located in `sample_pdfs/` directory (sample1.pdf, sample2.pdf, sample3.pdf).

## Architecture

### Modular Design
The application is organized into separate modules for maintainability and testability:

```
minimal-local-RAG/
├── main.py              # Entry point
├── config.py            # Configuration management
├── database.py          # ChromaDB operations
├── embeddings.py        # Embedding generation
├── pdf_processor.py     # PDF to structured data pipeline
├── rag.py               # Search and chat logic
├── cli.py               # Command-line interface
├── tests/               # Unit tests
│   ├── test_config.py
│   ├── test_pdf_processor.py
│   └── README.md
├── sample_pdfs/         # Sample LAQ PDFs
├── requirements.txt
├── .env.example         # Environment configuration template
├── CLAUDE.md           # This file
└── README.md
```

### Core Components

#### 1. Configuration (`config.py`)
- **Config dataclass** with environment variable support via `python-dotenv`
- Validates all settings on initialization (thresholds, top-k values, temperature)
- Creates database directory automatically
- All configurable parameters are documented in `.env.example`

#### 2. Database (`database.py`)
- **LAQDatabase class** encapsulates all ChromaDB operations
- Batch insertion support for better performance
- Duplicate ID detection before insertion
- Relevance filtering based on similarity threshold
- Custom exceptions: `DatabaseError`

#### 3. Embeddings (`embeddings.py`)
- **EmbeddingService class** handles embedding generation
- Verifies Ollama connection on initialization
- Batch embedding support for multiple texts
- Custom exceptions: `EmbeddingError`, `OllamaConnectionError`, `OllamaModelNotFoundError`
- Provides actionable error messages (e.g., "Run: ollama pull nomic-embed-text")

#### 4. PDF Processor (`pdf_processor.py`)
- **Pydantic models** for data validation: `QAPair`, `LAQData`
- **PDFProcessor class** handles PDF → structured data pipeline
- File validation (existence, extension, size warnings)
- JSON extraction with fallback regex parsing
- Automatic schema validation via Pydantic
- Custom exception: `PDFProcessingError`

#### 5. RAG Service (`rag.py`)
- **RAGService class** provides search and chat capabilities
- Improved chat prompt with clear instructions and citation requirements
- Context building with proper formatting and LAQ separation
- Match quality statistics and color-coded relevance indicators
- Configurable temperature (0.1) for factual responses

#### 6. CLI (`cli.py`)
- **CLI class** provides interactive menu-driven interface
- Progress bars using `tqdm` for long operations (embedding generation)
- Enhanced error handling with user-friendly messages
- Database info display
- Formatted output for search results and extracted data

#### 7. Main Entry Point (`main.py`)
- Simple entry point that initializes Config and CLI
- Comprehensive error handling with specific exception types
- Helpful error messages for common issues (Ollama not running, models not found)

### Core Pipeline Flow

1. **PDF Upload Workflow:**
   - User provides PDF path
   - `PDFProcessor.validate_pdf_file()` checks file existence, type, and size
   - `PDFProcessor.extract_markdown_from_pdf()` converts PDF → Markdown via Docling
   - `PDFProcessor.structure_laqs_with_mistral()` extracts structured JSON with Pydantic validation
   - Display extracted data to user for review
   - `EmbeddingService.embed_qa_pairs()` generates embeddings with progress bar
   - `LAQDatabase.store_qa_pairs()` performs batch insertion into ChromaDB

2. **Search Workflow:**
   - User enters query
   - `EmbeddingService.embed_text()` generates query embedding
   - `RAGService.search()` retrieves relevant LAQs with relevance filtering
   - Results displayed with match quality stats and color-coded indicators

3. **Chat Workflow:**
   - User enters question
   - `RAGService.chat()` retrieves top-k LAQs and builds formatted context
   - Improved prompt instructs LLM to cite sources and acknowledge missing information
   - Response generated with low temperature (0.1) for factual accuracy
   - Source LAQs displayed with similarity scores

### Data Storage

**ChromaDB Collection:** `laqs`
- **Path:** Configurable via `DB_PATH` (default: `./laq_db`)
- **Similarity metric:** Cosine
- **Document format:** `"Q: {question}\nA: {answer}"`
- **Metadata fields:** pdf, pdf_title, laq_num, qa_pair_num, type, question, answer, minister, date, attachments
- **Document IDs:** `{pdf_stem}_{laq_number}_qa{index}`

### Pydantic Models

**QAPair:**
```python
class QAPair(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
```

**LAQData:**
```python
class LAQData(BaseModel):
    pdf_title: str
    laq_type: str
    laq_number: str
    minister: str
    date: str
    qa_pairs: List[QAPair] = Field(min_items=1)
    tabled_by: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)
```

All LAQ data is validated against these schemas, ensuring data quality.

### LLM Prompt Engineering

#### Extraction Prompt (`pdf_processor.py`)
Detailed prompt for extracting structured LAQ data with:
- Clear output format specification with example
- Rules for handling sub-questions (a), (b), (c)
- Instructions to preserve exact wording
- JSON-only output requirement

#### Chat Prompt (`rag.py`)
Improved conversational prompt with:
- System instructions defining the assistant's role
- Context section with formatted LAQs
- 7 specific instructions including:
  - Answer only from provided context
  - Cite LAQ numbers explicitly
  - Acknowledge missing information
  - Maintain professional tone
  - Reference attachments when mentioned

### Error Handling

All modules use custom exception hierarchies:
- `DatabaseError` - Database operation failures
- `EmbeddingError` - Embedding generation failures
  - `OllamaConnectionError` - Cannot connect to Ollama
  - `OllamaModelNotFoundError` - Model not available
- `PDFProcessingError` - PDF processing failures
- `RAGError` - Search and chat failures

Error messages are actionable and guide users to solutions (e.g., "Run: ollama serve").

### Configuration

All settings can be customized via environment variables (see `.env.example`):
- Model selection (LLM and embedding models)
- Retrieval parameters (top-k, similarity threshold)
- Processing limits (chunk size, metadata length)
- LLM generation parameters (temperature, top-p)
- Ollama connection settings

### Match Quality Scoring
Search results use distance-to-similarity conversion: `score = (1 - distance) * 100`
- 🟢 80%+: Strong match
- 🟡 60-79%: Moderate match
- 🔴 <60%: Weak match

### Testing

Unit tests are located in `tests/`:
- `test_config.py` - Configuration validation tests
- `test_pdf_processor.py` - Pydantic model and PDF validation tests

Run tests with `pytest tests/` (requires `pytest` and `pytest-cov`).

### Key Improvements Over Original

1. **Modularity:** Split into 7 focused modules instead of single 400-line file
2. **Type Safety:** Pydantic models for data validation
3. **Error Handling:** Custom exception hierarchy with actionable messages
4. **Configuration:** Environment variable support with validation
5. **User Experience:** Progress bars, better formatting, database info display
6. **RAG Quality:** Improved prompts with citations and factual instructions
7. **Performance:** Batch database operations, duplicate detection
8. **Testability:** Dependency injection, unit tests, no global state
9. **Documentation:** Type hints, docstrings, comprehensive error messages

## Development Considerations

- ChromaDB persistence ensures data survives application restarts
- Markdown truncated to configurable limit (default: 10,000 chars) to avoid token limits
- Ollama must be running locally before starting the application
- No external API calls or internet connectivity required
- All configuration is centralized in `config.py` and can be overridden via `.env`
- Tests require Ollama to be running for integration tests (can be skipped)
