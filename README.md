# LAQ RAG System - Production Web Application

A minimal, privacy-friendly Retrieval-Augmented Generation (RAG) system that runs entirely locally without external API calls. The system processes Legislative Assembly Question (LAQ) PDFs, extracts Q&A pairs using LLMs, stores them in a vector database, and enables semantic search and chat interactions through a modern web interface.

## 🏗️ Architecture

```
minimal-local-RAG/
├── frontend/              # React + Vite web application
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Page components
│   │   ├── services/     # API service layer
│   │   └── styles/       # CSS styles
│   └── package.json
│
├── backend/              # FastAPI REST API
│   ├── app/
│   │   ├── api/         # API endpoints
│   │   ├── core/        # Configuration
│   │   ├── models/      # Pydantic schemas
│   │   └── services/    # Business logic
│   └── requirements.txt
│
├── docs/                 # Documentation
│   ├── CLAUDE.md        # Development guide
│   ├── DESIGN_GUIDE.md  # Design system
│   └── STYLE_GUIDE.md   # UI style guide
│
└── sample_pdfs/         # Sample LAQ PDFs
```

## 🚀 Tech Stack

### Backend
- **Framework:** FastAPI (async, high-performance)
- **LLM:** Mistral (via Ollama)
- **Embeddings:** nomic-embed-text (via Ollama)
- **Vector DB:** ChromaDB (cosine similarity)
- **Document Processing:** Docling (PDF to Markdown)
- **Validation:** Pydantic

### Frontend
- **Framework:** React 18
- **Build Tool:** Vite (fast, modern)
- **Routing:** React Router v6
- **HTTP Client:** Axios
- **Styling:** Vanilla CSS (design system)

## 📋 Prerequisites

### Required
1. **Python 3.10+**
2. **Node.js 18+** and npm
3. **Ollama** with required models:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

### Verify Installation
```bash
# Check Python
python --version

# Check Node.js
node --version

# Check Ollama
ollama list
```

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd minimal-local-RAG
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp ../.env.example .env

# Edit .env if needed (optional)
```

### 3. Frontend Setup
```bash
cd ../frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Edit .env if needed (optional - default: http://localhost:8000)
```

## 🎯 Running the Application

### Start Backend (Terminal 1)
```bash
cd backend
source venv/bin/activate  # Activate venv
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at:
- **API:** http://localhost:8000
- **Swagger Docs:** http://localhost:8000/api/docs
- **ReDoc:** http://localhost:8000/api/redoc

### Start Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```

Frontend will be available at:
- **Web App:** http://localhost:5173

### Verify Everything Works
1. Open http://localhost:5173
2. You should see the LAQ RAG Dashboard
3. Check database stats on the dashboard

## 📚 API Endpoints

### Upload
- `POST /api/upload/` - Upload and process LAQ PDF

### Search
- `POST /api/search/` - Semantic search on LAQs
  ```json
  {
    "query": "education budget",
    "top_k": 5,
    "threshold": 0.6
  }
  ```

### Chat
- `POST /api/chat/` - Chat with LAQ knowledge base
  ```json
  {
    "question": "What is the education budget?",
    "top_k": 5
  }
  ```

### Database
- `GET /api/database/info` - Get database information

### Health
- `GET /api/health` - Health check endpoint

## 🎨 Features

### ✅ Implemented
- 🏠 **Dashboard:** Overview with database stats
- 🔍 **Search:** Semantic search with similarity scores
- 📊 **Database Info:** View collection statistics
- 🎨 **Dark Mode UI:** Following Perplexity Pro design system
- 📱 **Responsive:** Mobile-friendly interface

### 🚧 Coming Soon
- 💬 **Chat Interface:** Full conversational UI
- 📤 **Drag & Drop Upload:** PDF upload with progress
- 📈 **Analytics:** Usage statistics and insights
- 🔐 **Authentication:** User management (optional)

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

### Frontend Tests
```bash
cd frontend
npm test  # Coming soon
```

## 📖 Usage Guide

### 1. Upload LAQ PDF
- Navigate to Upload page
- Select PDF file
- System will:
  - Extract Q&A pairs
  - Generate embeddings
  - Store in vector database

### 2. Search LAQs
- Navigate to Search page
- Enter search query
- View results with similarity scores
- See metadata (LAQ number, minister, date)

### 3. Chat with LAQs
- Navigate to Chat page
- Ask questions in natural language
- Get answers with source citations

### 4. View Database Info
- Navigate to Database page
- See collection statistics
- View configuration

## 🔧 Configuration

### Backend (.env)
```bash
# Database
DB_PATH=./laq_db
COLLECTION_NAME=laqs

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text

# RAG Settings
TOP_K=5
SIMILARITY_THRESHOLD=0.6
TEMPERATURE=0.1
```

### Frontend (.env)
```bash
VITE_API_BASE_URL=http://localhost:8000
```

## 📁 Project Structure Details

### Backend Services
- **config.py:** Configuration management
- **database.py:** ChromaDB operations
- **embeddings.py:** Embedding generation
- **pdf_processor.py:** PDF processing pipeline
- **rag.py:** Search and chat logic

### Frontend Pages
- **Dashboard:** Overview and quick actions
- **Search:** Semantic search interface
- **Chat:** Conversational interface (coming soon)
- **Upload:** PDF upload (coming soon)
- **Database:** Database information (coming soon)

## 🎨 Design System

The application follows a comprehensive design system inspired by Perplexity Pro:

- **Colors:** OKLCH color space, dark mode first
- **Typography:** System fonts, optimized weights
- **Spacing:** 8px grid system
- **Components:** Reusable, accessible UI components

See `docs/DESIGN_GUIDE.md` for complete design specifications.

## 🚀 Deployment (Future)

### Backend
```bash
# Production server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend
```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

Serve the `dist/` folder with any static file server.

## 🐛 Troubleshooting

### Ollama Connection Error
```bash
# Ensure Ollama is running
ollama serve

# Check models are installed
ollama list
```

### Backend Import Errors
```bash
# Ensure you're in the backend directory
cd backend

# Ensure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend Build Errors
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### CORS Errors
- Ensure backend is running on port 8000
- Check CORS configuration in `backend/app/main.py`
- Verify `VITE_API_BASE_URL` in frontend `.env`

## 📝 Development Notes

### Adding New API Endpoint
1. Create endpoint in `backend/app/api/endpoints/`
2. Add router to `backend/app/main.py`
3. Update frontend service in `frontend/src/services/api.js`
4. Create UI component in frontend

### Adding New Page
1. Create component in `frontend/src/pages/`
2. Add route in `frontend/src/App.jsx`
3. Add navigation link in `Sidebar.jsx`

## 🤝 Contributing

1. Follow the design guide (`docs/DESIGN_GUIDE.md`)
2. Maintain code style consistency
3. Write tests for new features
4. Update documentation

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- **Perplexity AI** - UI/UX inspiration
- **Ollama** - Local LLM infrastructure
- **ChromaDB** - Vector database
- **FastAPI** - Modern Python web framework
- **React** - UI library

## 📞 Support

For issues and questions:
- Check `docs/CLAUDE.md` for development guide
- See API docs at http://localhost:8000/api/docs
- Review design guide in `docs/DESIGN_GUIDE.md`

---

**Built with ❤️ for privacy-focused local RAG**
