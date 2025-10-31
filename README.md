
# Local Retrieval-Augmented Generation (RAG) with a Simple CLI Interface

A minimal, privacy-friendly Retrieval-Augmented Generation (RAG) system that runs **entirely on your local machine** — no external API calls or cloud dependencies.
This project lets you upload documents, embed them into a local vector database, and chat with them through a simple menu-driven Command Line Interface (CLI).


#### 🤖 What is RAG?

RAG stands for Retrieval-Augmented Generation — a technique that combines information retrieval with language generation to produce more accurate and context-aware responses.


In simple words, 
Instead of relying only on what the model was trained on, RAG allows it to look up relevant information from an external knowledge source (like your documents) before answering.


## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Language Model (LLM)** | [Mistral](https://mistral.ai) via [Ollama](https://ollama.ai) |
| **Embeddings Model** | `nomic-embed-text` |
| **Vector Database** | [ChromaDB](https://www.trychroma.com) |
| **Document Conversion** | [Docling](https://pypi.org/project/docling/) |
| **Language** | Python 3.10+ |
| **Interface** | Menu-driven CLI |

### 🧩 Supporting Libraries

- **`os`** and **`pathlib`** → for file handling and directory management  
- **`re`** → for regular expression–based text cleaning and preprocessing  





## 🧾 Installation

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/jeetsahoo/local-RAG-cli.git
cd local-RAG-cli
```

#### 2️⃣ Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Run Ollama in the Background

Make sure Ollama is installed and running:

```bash
ollama run llama3
```

#### 5️⃣ Launch the CLI

```bash
python main.py
```




## 🚀 How to Run

### 1. Run the Main Program

```bash
python main.py
```

You’ll see the following interactive menu:

```
====================================================================================================
LAQ RAG PIPELINE
====================================================================================================
1. Upload PDF
2. Search LAQ
3. Chat with LAQ
4. Clear Database
5. Exit
====================================================================================================
Select (1-5):
```


## 🧭 Menu Options Explained

### **1️⃣ Upload PDF**

1. Enter the file path to a PDF (e.g. `assets/sample_pdfs/question1.pdf`).
2. The system will:

   * Convert the **PDF → Markdown** using **Docling**.
   * Send the markdown to **Mistral LLM** (via **Ollama**).
   * Mistral structures the text into a clean JSON format:

     ```json
     {
       "pdf_title": "Road Development Projects",
       "laq_type": "Starred",
       "laq_number": "QA-324",
       "minister": "Mr. X",
       "date": "2024-07-10",
       "qa_pairs": [
         {"question": "What is the budget?", "answer": "Rs. 50 crores allocated."}
       ],
       "attachments": ["Annexure-A.pdf"]
     }
     ```
   * The structured Q&A pairs are embedded using **nomic-embed-text** and stored in **ChromaDB**.
3. You’ll be asked:

   ```
   ✅ Store this data in database? (yes/no):
   ```
4. Type `yes` to confirm saving.

At the end, you’ll see something like:

```
✅ Stored 5/5 Q&A pairs from question1.pdf
```

---

### **2️⃣ Search LAQ**

**Purpose:** Find relevant questions and answers semantically.

**How it works:**

1. Enter a natural language query, for example:

   ```
   Enter query: road construction projects in 2024
   ```
2. The system:

   * Embeds your query.
   * Searches similar embeddings in **ChromaDB**.
   * Displays the top 5 matches with metadata, confidence score, and context.

**Example Output:**

```
📍 SOURCE: ROAD_DEVELOPMENT_PROJECTS
LAQ #QA-324 (Starred) | Date: 2024-07-10 | 🟢 STRONG MATCH (89.4%)

👤 Minister: Mr. X

❓ QUESTION:
   What is the budget for road construction projects this year?

✅ ANSWER:
   Rs. 50 crores have been allocated for new road projects.
```

---

### **3️⃣ Chat with LAQ**

**Purpose:** Have a conversational interaction with the retrieved LAQs.

**How it works:**

1. Type a question, for example:

   ```
   Enter query: Tell me about the funds for road projects this year
   ```
2. The system retrieves the most relevant LAQs, builds a context, and sends them to **Mistral** to generate a contextual, human-like answer.

**Example Output:**

```
AI RESPONSE:
The Minister stated that Rs. 50 crores were allocated for new road construction projects in 2024.
Additional details are in Annexure-A.
```

💡 *This step performs the actual RAG process — combining retrieval (ChromaDB context) with generation (Mistral LLM).*

---

### **4️⃣ Clear Database (Optional)**

**Purpose:** Reset your **ChromaDB** collection.
Useful when testing new PDFs or starting from scratch.




### **5️⃣ Exit**

Simply quits the application.





## 📦 Directory Structure

```
local-RAG-cli/
├── main.py                 # Main script file
├── sample_pdfs/            # Sample documents
├── laq_db/                 # Local ChromaDB database
├── requirements.txt        # Dependencies
└── README.md              
```

---

## 💬 Future Enhancements

* 🌐 Web-based interface using Streamlit
* 🧾 Optimise Chunk size
* 🔄 Multi document ingestion support
* 🗂️ Multi-user support
* 🧠 Embedding model selection from CLI
* 📦 Deploy using Docker


---

## 🧑‍💻 Author

**Jitendra Sahoo**

🔗 [LinkedIn](http://www.linkedin.com/in/jitendra-sahoo-31a187265)


