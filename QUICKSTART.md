# Quick Start Guide

## 🚀 Get Up and Running in 5 Minutes

### Prerequisites Check
```bash
# 1. Python 3.10+
python --version

# 2. Node.js 18+
node --version

# 3. Ollama with models
ollama list
# Should show: mistral, nomic-embed-text
# If not: ollama pull mistral && ollama pull nomic-embed-text
```

### Installation (One-Time Setup)

```bash
# 1. Backend Setup
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac (Windows: venv\Scripts\activate)
pip install -r requirements.txt
cd ..

# 2. Frontend Setup
cd frontend
npm install
cd ..
```

### Running the App

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```
✅ Backend running at http://localhost:8000

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```
✅ Frontend running at http://localhost:5173

### Access the App
Open your browser: **http://localhost:5173**

---

## 📋 Project Structure at a Glance

```
minimal-local-RAG/
├── backend/          → FastAPI server (port 8000)
│   └── app/
│       ├── main.py   → Entry point
│       ├── api/      → REST endpoints
│       └── services/ → Your original Python modules
│
├── frontend/         → React app (port 5173)
│   └── src/
│       ├── App.jsx   → Main app
│       ├── pages/    → Dashboard, Search, Chat, etc.
│       └── services/ → API calls to backend
│
├── docs/            → Design guides & documentation
└── sample_pdfs/     → Test PDFs
```

---

## 🧪 Quick Test

### 1. Check Backend API
Visit: http://localhost:8000/api/docs

### 2. Test Search (if you have data)
1. Go to http://localhost:5173/search
2. Enter a query
3. See results!

### 3. View Database Stats
1. Go to http://localhost:5173
2. Check the stat cards

---

## ❓ Troubleshooting

### Ollama Not Connected
```bash
# Start Ollama
ollama serve

# Verify models
ollama list
```

### Backend Won't Start
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend Won't Start
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### CORS Errors
- Make sure backend is on port **8000**
- Make sure frontend is on port **5173**

---

## 📚 Next Steps

1. **Read the full README:** `README.md`
2. **Check design guide:** `docs/DESIGN_GUIDE.md`
3. **Review API docs:** http://localhost:8000/api/docs
4. **Upload a PDF:** Go to Upload page (coming soon)

---

## 🎯 Common Tasks

### Add Sample Data
```bash
# Use the original CLI (if needed)
python main.py  # From root directory
```

### View Logs
- **Backend:** Check terminal 1
- **Frontend:** Check terminal 2
- **Browser:** Open DevTools console

### Stop Servers
- Press `Ctrl+C` in both terminals

---

## 📞 Need Help?

- **API Documentation:** http://localhost:8000/api/docs
- **Design System:** `docs/DESIGN_GUIDE.md`

**Happy coding! 🎉**
