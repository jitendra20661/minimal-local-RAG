# LAQ RAG System - Project Overview

## 🎯 What This Is

A **privacy-focused, locally-running** web application for searching and chatting with Legislative Assembly Question (LAQ) documents using AI/RAG technology.

**Key Features:**
- 🔒 **100% Local** - No external APIs, all processing on your machine
- 🤖 **AI-Powered** - Uses Mistral LLM and semantic search
- 🎨 **Modern UI** - Clean, dark-mode interface inspired by Perplexity Pro
- ⚡ **Fast** - React + Vite frontend, FastAPI backend

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   USER BROWSER                      │
│              http://localhost:5173                  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │         React Frontend (Vite)                │  │
│  │  - Dashboard, Search, Chat, Upload pages     │  │
│  │  - Dark mode UI (Perplexity Pro style)       │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────┘
                     │ HTTP Requests (Axios)
                     │ API calls to backend
                     ↓
┌─────────────────────────────────────────────────────┐
│              FastAPI Backend Server                 │
│              http://localhost:8000                  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  REST API Endpoints:                         │  │
│  │  • POST /api/upload/     → Upload PDF        │  │
│  │  • POST /api/search/     → Search LAQs       │  │
│  │  • POST /api/chat/       → Chat with LAQs    │  │
│  │  • GET  /api/database/   → DB info           │  │
│  └──────────────────────────────────────────────┘  │
└─────┬───────────────────────┬───────────────────────┘
      │                       │
      │                       ↓
      │            ┌──────────────────────┐
      │            │   Ollama (Local)     │
      │            │  - Mistral LLM       │
      │            │  - nomic-embed-text  │
      │            └──────────────────────┘
      ↓
┌──────────────────────┐
│   ChromaDB (Local)   │
│   Vector Database    │
│   - LAQ embeddings   │
│   - Cosine search    │
└──────────────────────┘
```

---

## 📁 Directory Structure

```
minimal-local-RAG/
│
├── backend/                    # Python FastAPI Server
│   ├── app/
│   │   ├── main.py            # FastAPI app entry
│   │   ├── api/               # REST endpoints
│   │   │   └── endpoints/
│   │   │       ├── upload.py  # PDF upload
│   │   │       ├── search.py  # Search endpoint
│   │   │       ├── chat.py    # Chat endpoint
│   │   │       └── database.py
│   │   ├── core/              # Configuration
│   │   │   └── config.py      # Settings
│   │   ├── models/            # Pydantic schemas
│   │   │   └── schemas.py     # Request/response models
│   │   └── services/          # Business logic
│   │       ├── config.py      # Config service
│   │       ├── database.py    # ChromaDB ops
│   │       ├── embeddings.py  # Embedding generation
│   │       ├── pdf_processor.py # PDF processing
│   │       └── rag.py         # RAG logic
│   ├── requirements.txt       # Python dependencies
│   └── tests/                 # Backend tests
│
├── frontend/                   # React + Vite App
│   ├── src/
│   │   ├── App.jsx            # Main app component
│   │   ├── main.jsx           # React entry point
│   │   ├── components/        # Reusable components
│   │   │   └── layout/
│   │   │       ├── Layout.jsx # App layout
│   │   │       ├── Sidebar.jsx # Navigation
│   │   │       └── Header.jsx
│   │   ├── pages/             # Page components
│   │   │   ├── Dashboard.jsx  # Home page
│   │   │   ├── Search.jsx     # Search interface
│   │   │   ├── Chat.jsx       # Chat interface
│   │   │   ├── Upload.jsx     # PDF upload
│   │   │   └── Database.jsx   # DB info
│   │   ├── services/          # API layer
│   │   │   └── api.js         # Axios API calls
│   │   └── styles/            # CSS
│   │       └── index.css      # Global styles
│   ├── package.json           # Node dependencies
│   ├── vite.config.js         # Vite config
│   └── index.html             # HTML entry
│
├── docs/                       # Documentation
│   ├── CLAUDE.md              # Dev guide (original)
│   ├── DESIGN_GUIDE.md        # Design system
│   ├── STYLE_GUIDE.md         # UI guidelines
│   └── dashboard_reference.html
│
├── sample_pdfs/               # Test PDFs
│   ├── sample1.pdf
│   ├── sample2.pdf
│   └── sample3.pdf
│
├── README.md                  # Main documentation
├── QUICKSTART.md              # Quick setup guide
├── MIGRATION_SUMMARY.md       # Restructuring details
└── .env.example               # Environment template
```

---

## 🔄 Data Flow

### 1. PDF Upload Flow
```
User → Upload PDF in browser
  ↓
Frontend sends file to /api/upload/
  ↓
Backend receives PDF
  ↓
PDFProcessor extracts text (Docling)
  ↓
LLM (Mistral) structures Q&A pairs
  ↓
EmbeddingService generates embeddings
  ↓
LAQDatabase stores in ChromaDB
  ↓
Success response to frontend
```

### 2. Search Flow
```
User → Types query in search box
  ↓
Frontend sends query to /api/search/
  ↓
Backend generates query embedding
  ↓
ChromaDB performs cosine similarity search
  ↓
RAGService filters and ranks results
  ↓
Results with scores returned to frontend
  ↓
User sees relevant LAQs
```

### 3. Chat Flow
```
User → Asks question in chat
  ↓
Frontend sends question to /api/chat/
  ↓
Backend retrieves relevant LAQs (search)
  ↓
RAGService builds context from top LAQs
  ↓
LLM (Mistral) generates answer with citations
  ↓
Answer + source LAQs returned to frontend
  ↓
User sees AI-generated answer with references
```

---

## 🛠️ Tech Stack Details

### Backend Stack
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | FastAPI | Async REST API |
| LLM | Mistral (Ollama) | Answer generation |
| Embeddings | nomic-embed-text | Vector generation |
| Vector DB | ChromaDB | Semantic search |
| PDF Parser | Docling | PDF → Markdown |
| Validation | Pydantic | Schema validation |
| Server | Uvicorn | ASGI server |

### Frontend Stack
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | React 18 | UI library |
| Build Tool | Vite | Fast dev server |
| Router | React Router v6 | Navigation |
| HTTP Client | Axios | API calls |
| Styling | Vanilla CSS | Custom design system |

---

## 🎨 Design System

Based on Perplexity Pro's interface:

**Visual Identity:**
- Dark mode primary (#100e12 background)
- Cyan accent color (#00d4c4)
- OKLCH color space
- System fonts

**Typography:**
- Base: 16px
- Weights: 375 (normal), 475 (medium), 575 (semibold)
- Letter spacing: 0.01em (dark mode optimized)

**Spacing:**
- 8px grid system
- Scale: 2xs(2px) → xs(4px) → sm(8px) → md(16px) → lg(32px) → xl(48px)

**Components:**
- Buttons: 8px border radius, 44px min height
- Cards: 12px border radius, subtle backgrounds
- Inputs: 8-12px radius, accent focus states
- Modals: 16px radius, backdrop blur

See `docs/DESIGN_GUIDE.md` for full specifications.

---

## 🚀 Quick Start

**1. Prerequisites:**
```bash
# Install Ollama
ollama pull mistral
ollama pull nomic-embed-text
```

**2. Setup:**
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

**3. Run:**
```bash
# Terminal 1
cd backend && uvicorn app.main:app --reload

# Terminal 2
cd frontend && npm run dev
```

**4. Access:**
- Frontend: http://localhost:5173
- API Docs: http://localhost:8000/api/docs

---

## 📊 Current Status

### ✅ Implemented
- Backend API with all endpoints
- Frontend structure with routing
- Dashboard page with database stats
- Search page with results display
- Dark mode UI following design system
- API documentation (Swagger/ReDoc)
- Comprehensive documentation

### 🚧 In Progress
- Chat page UI
- Upload page with drag-and-drop
- Database info page

### 🔮 Planned
- Analytics dashboard
- Advanced search filters
- Export functionality
- User preferences
- Mobile optimization

---

## 📚 Documentation Guide

**For getting started:**
→ `QUICKSTART.md` - 5-minute setup

**For understanding the system:**
→ `README.md` - Full documentation
→ `MIGRATION_SUMMARY.md` - What changed

**For development:**
→ `docs/CLAUDE.md` - Original dev guide
→ `docs/DESIGN_GUIDE.md` - Design patterns & rules
→ `docs/STYLE_GUIDE.md` - Perplexity Pro reference

**For API reference:**
→ http://localhost:8000/api/docs - Interactive API docs

---

## 🔧 Configuration

**Backend (`.env` or environment variables):**
```bash
DB_PATH=./laq_db
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text
TOP_K=5
SIMILARITY_THRESHOLD=0.6
TEMPERATURE=0.1
```

**Frontend (`.env`):**
```bash
VITE_API_BASE_URL=http://localhost:8000
```

---

## 🎯 Use Cases

1. **Legislative Research**
   - Search thousands of LAQs instantly
   - Find relevant questions by topic
   - Get AI-powered summaries

2. **Policy Analysis**
   - Ask questions about government policy
   - Get answers with source citations
   - Track minister responses

3. **Knowledge Management**
   - Centralized LAQ repository
   - Semantic search capabilities
   - Historical question tracking

---

## 🔒 Privacy & Security

**Local-First:**
- ✅ No external API calls
- ✅ All data stays on your machine
- ✅ No internet required (after setup)
- ✅ Complete data ownership

**Security Notes:**
- All LLM processing local (Ollama)
- ChromaDB stored locally
- No telemetry or tracking
- Optional: Add authentication for multi-user

---

## 🤝 Contributing

See development guides:
1. Follow design system (`docs/DESIGN_GUIDE.md`)
2. Use existing components where possible
3. Write tests for new features
4. Update documentation

---

## 📞 Support

**Documentation:**
- Main: `README.md`
- Quick: `QUICKSTART.md`
- Design: `docs/DESIGN_GUIDE.md`

**API Docs:**
- http://localhost:8000/api/docs

**Troubleshooting:**
- Check Ollama is running: `ollama list`
- Check ports: Backend 8000, Frontend 5173
- Check logs in terminal outputs

---

**Built with ❤️ for local, privacy-focused RAG**
