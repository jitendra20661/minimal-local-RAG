# Minimal Local RAG System

A minimal, privacy-friendly **Retrieval-Augmented Generation (RAG)** system that runs **entirely locally** without external API calls. Built specifically for processing Legislative Assembly Question (LAQ) PDFs with semantic search and conversational chat capabilities.

## What is RAG?

RAG (Retrieval-Augmented Generation) combines information retrieval with language generation. Instead of relying only on what an LLM was trained on, RAG retrieves relevant information from your documents before generating answers—resulting in more accurate, context-aware, and factual responses.

## Features

- **100% Local & Private** - No external API calls, all processing happens on your machine
- **Modular Architecture** - Clean separation of concerns across 7 focused modules
- **Type-Safe** - Pydantic models ensure data validation and schema compliance
- **Progress Tracking** - Visual feedback with progress bars for long operations
- **Smart Search** - Semantic search with relevance scoring and color-coded match quality
- **Conversational Chat** - RAG-powered chat with source citations
- **Configurable** - Environment variable support via `.env` file
- **Tested** - Unit tests with pytest for core components
- **Error Handling** - Actionable error messages guide you to solutions

## Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | [Mistral](https://mistral.ai) via [Ollama](https://ollama.ai) |
| **Embeddings** | `nomic-embed-text` via Ollama |
| **Vector Database** | [ChromaDB](https://www.trychroma.com) with cosine similarity |
| **Document Processing** | [Docling](https://github.com/DS4SD/docling) (PDF → Markdown) |
| **Data Validation** | [Pydantic](https://docs.pydantic.dev/) |
| **CLI Interface** | Python with tqdm progress bars |
| **Language** | Python 3.10+ |

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Maxwell-Fernandes/minimal-local-RAG.git
cd minimal-local-RAG
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama

**Install Ollama:**
- Download from [https://ollama.ai](https://ollama.ai)
- Follow installation instructions for your OS

**Pull Required Models:**
```bash
# Start Ollama (if not auto-started)
ollama serve

# Pull models (in a new terminal)
ollama pull mistral
ollama pull nomic-embed-text
```

### 5. (Optional) Configure Environment

```bash
cp .env.example .env
# Edit .env to customize settings (model names, thresholds, etc.)
```

### 6. Run the Application

```bash
python main.py
```

## Project Structure

```
minimal-local-RAG/
├── main.py              # Entry point
├── config.py            # Configuration management with environment variables
├── database.py          # ChromaDB operations (batch insert, search, filtering)
├── embeddings.py        # Embedding generation with Ollama
├── pdf_processor.py     # PDF → structured data pipeline with Pydantic
├── rag.py               # Search and chat logic
├── cli.py               # Interactive command-line interface
├── tests/               # Unit tests
│   ├── test_config.py
│   ├── test_pdf_processor.py
│   └── README.md
├── sample_pdfs/         # Sample LAQ PDFs for testing
├── requirements.txt     # Python dependencies
├── .env.example         # Environment configuration template
├── CLAUDE.md           # Developer documentation
└── README.md           # This file
```

## How It Works

### 1. PDF Upload Pipeline

```
PDF File → Docling (PDF→Markdown) → Mistral LLM (Extract Structure)
→ Pydantic Validation → Embedding Generation → ChromaDB Storage
```

1. **File Validation**: Checks file exists, has .pdf extension, warns on large files
2. **Markdown Extraction**: Docling converts PDF to structured markdown
3. **LLM Structuring**: Mistral extracts Q&A pairs into validated JSON schema
4. **Data Validation**: Pydantic models ensure data quality and completeness
5. **Embedding**: nomic-embed-text generates vector embeddings with progress bars
6. **Storage**: Batch insertion into ChromaDB with duplicate detection

### 2. Search Workflow

```
User Query → Embedding → ChromaDB Similarity Search
→ Relevance Filtering → Ranked Results Display
```

- Semantic search using cosine similarity
- Match quality scoring: 🟢 Strong (80%+), 🟡 Moderate (60-79%), 🔴 Weak (<60%)
- Displays full context: question, answer, minister, date, attachments

### 3. Chat Workflow

```
User Question → Retrieve Top-K LAQs → Build Context
→ Mistral Generation (Low Temperature) → Answer with Citations
```

- Retrieves most relevant LAQs as context
- Instructs LLM to cite sources and acknowledge missing information
- Low temperature (0.1) for factual, accurate responses

## Usage

### Main Menu

```
====================================================================================================
LAQ RAG SYSTEM
====================================================================================================
1. Upload PDF           - Process and store LAQ documents
2. Search LAQs          - Semantic search across stored documents
3. Chat with LAQs       - Conversational Q&A with source citations
4. Database Info        - View collection statistics
5. Clear Database       - Reset and start fresh
6. Exit
====================================================================================================
```

### Example: Upload PDF

```bash
Select (1-6): 1
Enter PDF path: sample_pdfs/sample1.pdf

🔄 Converting PDF to Markdown...
✅ Conversion successful
🤖 Processing with Mistral LLM...

📊 STRUCTURED LAQ DATA
====================================================================================================
📄 PDF Title: TENDER ISSUED FOR LEASING OF JETTY SPACE
📝 LAQ Type: Starred
🔢 LAQ Number: 010C
👤 Minister: Shri. Aleixo Sequeira
📅 Date: 08-08-2024

❓ Question-Answer Pairs: 3
[Shows extracted Q&A pairs...]

✅ Store this data in database? (yes/no): yes
Generating embeddings: 100%|████████████████████| 3/3 [00:02<00:00, 1.2it/s]
✅ Stored 3/3 Q&A pairs from sample1.pdf
```

### Example: Search

```bash
Select (1-6): 2
Enter query: jetty leasing tender details

🔍 SEARCH RESULTS
====================================================================================================
RESULT #1
📍 SOURCE: SAMPLE1
   LAQ #010C (Starred) | Date: 08-08-2024 | 🟢 STRONG MATCH (87.3%)

❓ QUESTION: the details of all tender issued for leasing jetty space...
✅ ANSWER: Santa Monica Jetty: Rs. 1.23 Cr plus taxes...
====================================================================================================
```

### Example: Chat

```bash
Select (1-6): 3
Enter question: What are the tender amounts for jetty leasing?

🤖 Generating response...

AI RESPONSE:
====================================================================================================
According to LAQ #010C, the Santa Monica Jetty tender (GTDC/JETTY/2019-20/3185)
had a financial bid of Rs. 1.23 Crores plus GST (Rs. 22,14,000).

The annual amounts received are:
- 2023-24: Rs. 1,45,14,000 (inclusive of GST)
- 2024-25: Rs. 1,45,14,000 (inclusive of GST)

Source: LAQ #010C (Starred), dated 08-08-2024
====================================================================================================

📚 Source LAQs:
  • LAQ #010C (87.3% match)
  • LAQ #025A (72.1% match)
```

## Configuration

Customize settings via `.env` file (see `.env.example`):

```bash
# Models
LLM_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text

# Retrieval Parameters
TOP_K_SEARCH=5
TOP_K_CHAT=3
SIMILARITY_THRESHOLD=0.3

# Processing Limits
MARKDOWN_MAX_LENGTH=10000
METADATA_MAX_LENGTH=500

# LLM Generation
TEMPERATURE=0.1
TOP_P=0.9

# Database
DB_PATH=./laq_db
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=. --cov-report=html

# View coverage
open htmlcov/index.html  # or your browser
```

## Sample PDFs

Three sample LAQ PDFs are included in `sample_pdfs/`:
- `sample1.pdf` - Jetty leasing tender questions
- `sample2.pdf` - Infrastructure development queries
- `sample3.pdf` - Budget allocation questions

## Architecture Highlights

### Modular Design
- **config.py**: Centralized configuration with validation
- **database.py**: ChromaDB operations with custom exceptions
- **embeddings.py**: Ollama embedding service with connection verification
- **pdf_processor.py**: PDF → structured data with Pydantic validation
- **rag.py**: Search and chat with improved prompts
- **cli.py**: User interface with progress tracking
- **main.py**: Entry point with comprehensive error handling

### Data Models (Pydantic)

```python
class QAPair(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)

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

### Error Handling

Custom exception hierarchy provides actionable guidance:
- `DatabaseError` - Database operation failures
- `EmbeddingError` - Embedding generation issues
  - `OllamaConnectionError` - "Run: ollama serve"
  - `OllamaModelNotFoundError` - "Run: ollama pull mistral"
- `PDFProcessingError` - PDF processing failures

## Troubleshooting

### Ollama Connection Error
```
❌ Error: Cannot connect to Ollama
Solution: Run 'ollama serve' in a terminal
```

### Model Not Found
```
❌ Error: Model 'mistral' not found
Solution: Run 'ollama pull mistral'
```

### No Search Results
- Check if PDFs have been uploaded and stored
- Try lowering `SIMILARITY_THRESHOLD` in `.env`
- Use more general search terms

## Future Enhancements

- [ ] Web interface with Streamlit or Gradio
- [ ] Support for additional document formats (DOCX, TXT, HTML)
- [ ] Advanced chunking strategies for long documents
- [ ] Multi-document ingestion in a single operation
- [ ] Query history and saved searches
- [ ] Export search results to CSV/JSON
- [ ] Docker deployment with docker-compose
- [ ] Support for multiple embedding models
- [ ] Incremental updates without re-processing

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [Ollama](https://ollama.ai) for local LLM inference
- PDF processing powered by [Docling](https://github.com/DS4SD/docling)
- Vector storage via [ChromaDB](https://www.trychroma.com)
- Original concept inspired by local RAG implementations

---

**Note**: This system requires Ollama to be installed and running locally. No external API keys or internet connectivity required for operation.
