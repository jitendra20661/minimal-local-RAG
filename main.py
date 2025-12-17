import os
import re
import json
from pathlib import Path
import chromadb
import ollama
from docling.document_converter import DocumentConverter
# from docling.document_converter import ConvertedDocument

LLM_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_0")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

DB_PATH = "./laq_db"
os.makedirs(DB_PATH, exist_ok=True)

# Initialize ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="laqs", metadata={"hnsw:space": "cosine"})


# ============= HELPER FUNCTIONS =============



def store_in_ChromaDB(laq_data, pdf_name):
    """Store Mistral-processed LAQ data in ChromaDB"""
    try:
        qa_pairs = laq_data.get('qa_pairs', [])
        if not qa_pairs:
            print("❌ No Q&A pairs to store")
            return
        
        success_count = 0
        for idx, qa in enumerate(qa_pairs, 1):
            try:
                laq_num = laq_data.get('laq_number', 'unknown')
                doc_id = f"{Path(pdf_name).stem}_{laq_num}_qa{idx}"
                
                question = qa.get('question', '')
                answer = qa.get('answer', '')
                text = f"Q: {question}\nA: {answer}"
                
                embedding = embed_text(text)
                if not embedding:
                    print(f"⚠️ Skipping Q&A {idx} - embedding failed")
                    continue
                
                metadata = {
                    "pdf": pdf_name,
                    "pdf_title": laq_data.get('pdf_title', 'N/A'),
                    "laq_num": str(laq_num),
                    "qa_pair_num": str(idx),
                    "type": laq_data.get('laq_type', 'N/A'),
                    "question": question[:500],
                    "answer": answer[:500],
                    "minister": laq_data.get('minister', 'N/A'),
                    "date": laq_data.get('date', 'N/A'),
                    "attachments": json.dumps(laq_data.get('attachments', []))
                }
                
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[text]
                )
                success_count += 1
            except Exception as e:
                print(f"⚠️ Error storing Q&A pair {idx}: {e}")
        
        print(f"✅ Stored {success_count}/{len(qa_pairs)} Q&A pairs from {pdf_name}")
    except Exception as e:
        print(f"❌ Storage error: {e}")



def embed_text(text):
    """Generate embedding"""
    try:
        return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        return []

# ============= MAIN FUNCTIONS =============

def extract_markdown_from_pdf(pdf_path):
    """Extract markdown from PDF using Docling"""
    try:
        print(f"🔄 Converting PDF to markdown...")
        converter = DocumentConverter()
        doc = converter.convert(pdf_path)
        markdown_data = doc.document.export_to_markdown()
        print(f"✅ Conversion successful")
        return markdown_data
    except Exception as e:
        print(f"❌ Failed to convert PDF to Md, error: {e}")
        return None


def structure_laqs_with_mistral(markdown_data: str, pdf_path: str):

    try:
        print("🤖 Processing LAQ Markdown with Mistral...")

        prompt = f"""
You are a structured data extraction assistant specialized in OFFICIAL Legislative Assembly Question (LAQ) documents.

IMPORTANT CONTEXT:
The input is RAW MARKDOWN TEXT extracted from an official LAQ PDF.
It may contain headings, bullet points, flattened tables, page headers/footers,
irregular line breaks, and sub-questions marked as (a), (b), (c), etc.

Your task is to extract COMPLETE, ACCURATE, and MACHINE-READABLE structured data.
This output will be consumed by an e-Governance system, so correctness is critical.

────────────────────────────────────────────
CORE TASKS
────────────────────────────────────────────

1. Extract LAQ metadata:
   - pdf_title
   - laq_type (Starred / Unstarred)
   - laq_number
   - minister
   - tabled_by
   - date

2. Extract QUESTION–ANSWER pairs:
   - EACH sub-question ((a), (b), (c), etc.) MUST be a SEPARATE entry.
   - Do NOT merge sub-questions.
   - Preserve original wording EXACTLY.
   - Do NOT paraphrase.
   - Questions and answers must be complete.

3. For EACH question–answer pair:
   - Identify ALL relevant GOVERNANCE DOMAINS.
   - Assign roles: "Primary" or "Secondary".
   - Limit to MAXIMUM 3 domains.
   - Assign confidence scores (0.00–1.00).
   - Map EACH domain to its official Demand Number.

4. Compute analytics fields:
   - total_domains_identified
   - is_inter_domain (true if more than one domain)

────────────────────────────────────────────
GOVERNANCE DOMAIN RULES
────────────────────────────────────────────

- Choose departments ONLY from the list below.
- Use department names EXACTLY as written.
- DO NOT invent new departments.
- Always choose ONE Primary domain when possible.
- If no domain clearly applies, return:

"domains": [
  {{
    "department": "Unclear",
    "demand_number": null,
    "role": "Primary",
    "confidence": 0.50
  }}
]

────────────────────────────────────────────
DEPARTMENT → DEMAND NUMBER MAPPING
────────────────────────────────────────────

Health → 32
Education → 27
Agriculture → 10
Water Resources → 22
Public Works Department → 21
Transport → 33
Tourism → 41
Revenue → 16
Urban Development → 25
Rural Development → 23
Power → 30
Environment → 19
Home → 18
Industries → 11
Ports → 40
River Navigation → 42
Fisheries → 12
Social Welfare → 35
Women and Child Development → 34
Housing → 26
Law and Judiciary → 17
Planning and Statistics → 08
Cooperation → 14
Information Technology → 05
Forest → 20
Mining → 09
Disaster Management → 36
Panchayati Raj → 24
Skill Development → 38
Labour and Employment → 15
Food and Civil Supplies → 13
Unclear → null

────────────────────────────────────────────
FEW-SHOT EXAMPLES
────────────────────────────────────────────

INPUT (Markdown):
(a) whether additional anganwadi centres have been sanctioned?
(b) steps taken to provide health check-ups to anganwadi children?

OUTPUT (partial):
{{
  "qa_pairs": [
    {{
      "question": "(a) whether additional anganwadi centres have been sanctioned?",
      "answer": "...",
      "domains": [
        {{
          "department": "Women and Child Development",
          "demand_number": 34,
          "role": "Primary",
          "confidence": 0.86
        }}
      ],
      "total_domains_identified": 1,
      "is_inter_domain": false
    }},
    {{
      "question": "(b) steps taken to provide health check-ups to anganwadi children?",
      "answer": "...",
      "domains": [
        {{
          "department": "Women and Child Development",
          "demand_number": 34,
          "role": "Primary",
          "confidence": 0.82
        }},
        {{
          "department": "Health",
          "demand_number": 32,
          "role": "Secondary",
          "confidence": 0.71
        }}
      ],
      "total_domains_identified": 2,
      "is_inter_domain": true
    }}
  ]
}}

────────────────────────────────────────────
REQUIRED OUTPUT FORMAT (STRICT JSON)
────────────────────────────────────────────

{{
  "pdf_title": "",
  "laq_type": "",
  "laq_number": "",
  "minister": "",
  "tabled_by": "",
  "date": "",
  "qa_pairs": [
    {{
      "question": "",
      "answer": "",
      "domains": [
        {{
          "department": "",
          "demand_number": null,
          "role": "Primary",
          "confidence": 0.00
        }}
      ],
      "total_domains_identified": 1,
      "is_inter_domain": false
    }}
  ],
  "attachments": []
}}

────────────────────────────────────────────
STRICT EXTRACTION RULES
────────────────────────────────────────────

1. Treat (a), (b), (c) as HARD boundaries.
2. Combine lines until next sub-question.
3. Preserve formatting using "\\n".
4. NEVER paraphrase or summarize.
5. Output ONLY valid JSON.
6. No explanations, comments, or markdown.

Now extract the structured data from the following MARKDOWN text:

{markdown_data[:12000]}
"""

        response = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            stream=False,
            format="json",
            options={"temperature": 0}
        )

        raw_text = response.get("response", "").strip()

        # Attempt strict JSON parsing
        try:
            return json.loads(raw_text)

        except json.JSONDecodeError:
            # Fallback: extract JSON block
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                print("❌ Failed to parse JSON from Mistral output")
                print(raw_text[:500])
                return None

    except Exception as e:
        print(f"❌ Mistral processing error: {e}")
        return None



def upload_pdf():
    """Upload and process PDF with Docling and Mistral"""
    path = input("Enter PDF path: ").strip()
    if not os.path.exists(path):
        print("❌ File not found")
        return
    
    pdf_name = Path(path).name
    
    # Step 1: Convert PDF to markdown with Docling
    markdown_data = extract_markdown_from_pdf(path)
    if not markdown_data:
        return
    
    # Step 2: Process markdown with Mistral LLM, extract structured QA pairs
    laq_data = structure_laqs_with_mistral(markdown_data, path)
    if not laq_data:
        return
    
    # Step 3: Display processed data
    print("\n" + "="*100)
    print("📊 STRUCTURED LAQ DATA (via Mistral)")
    print("="*100)
    print(laq_data)
    print(f"\n📄 PDF Title: {laq_data.get('pdf_title', 'N/A')}")
    print(f"📝 LAQ Type: {laq_data.get('laq_type', 'N/A')}")
    print(f"🔢 LAQ Number: {laq_data.get('laq_number', 'N/A')}")
    print(f"👤 Minister: {laq_data.get('minister', 'N/A')}")
    print(f"📅 Date: {laq_data.get('date', 'N/A')}")
    
    qa_pairs = laq_data.get('qa_pairs', [])
    print(f"\n❓ Question-Answer Pairs: {len(qa_pairs)}")
    
    print("\n" + "─"*100)
    for idx, qa in enumerate(qa_pairs, 1):
        print(f"\n[Q&A Pair {idx}]")
        print(f" ❓ Q: {qa.get('question', 'N/A')}")
        print(f" ✅ A: {qa.get('answer', 'N/A')}")
    
    attachments = laq_data.get('attachments', [])
    if attachments:
        print(f"\n📎 Attachments:")
        for att in attachments:
            print(f" • {att}")
    
    print("\n" + "="*100)
    
    # Step 4: Store in ChromaDB
    confirm = input("\n✅ Store this data in database? (yes/no): ").strip().lower()
    if confirm == "yes":
        store_in_ChromaDB(laq_data, pdf_name)
    else:
        print("❌ Data not stored")

def search_laq():
    query = input("Enter query: ").strip()
    if not query:
        print("❌ Query cannot be empty")
        return
    
    try:
        query_embedding = embed_text(query)
        if not query_embedding:
            print("❌ Could not generate embedding for query")
            return
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["distances", "metadatas"]
        )
        
        if not results['ids'] or not results['ids'][0]:
            print("❌ No results found")
            return
        
        print("\n" + "="*100)
        print("🔍 SEARCH RESULTS FOR: " + query.upper())
        print("="*100)
        
        for i, doc_id in enumerate(results['ids'][0]):
            dist = results['distances'][0][i]
            score = round((1 - dist) * 100, 2)
            meta = results['metadatas'][0][i]
            
            pdf_name = meta.get('pdf', 'Unknown').replace('.pdf', '').upper()
            laq_type = meta.get('type', 'N/A')
            laq_no = meta.get('laq_num', 'N/A')
            
            # Color-coded match quality
            if score >= 80:
                match_quality = "🟢 STRONG MATCH"
            elif score >= 60:
                match_quality = "🟡 MODERATE MATCH"
            else:
                match_quality = "🔴 WEAK MATCH"
            
            print(f"\n┌{'─'*98}┐")
            print(f"│ RESULT #{i+1} {' '*85}│")
            print(f"└{'─'*98}┘")
            
            print(f"\n📍 SOURCE: {pdf_name}")
            print(f" LAQ #{laq_no} ({laq_type}) | Date: {meta.get('date', 'N/A')} | {match_quality} ({score}%)")
            
            print(f"\n👤 Minister: {meta.get('minister', 'Not mentioned')}")
            
            print(f"\n❓ QUESTION:\n {meta.get('question', 'N/A')[:200]}..." if len(meta.get('question', '')) > 200 else f"\n❓ QUESTION:\n {meta.get('question', 'N/A')}")
            
            print(f"\n✅ ANSWER:\n {meta.get('answer', 'N/A')[:200]}..." if len(meta.get('answer', '')) > 200 else f"\n✅ ANSWER:\n {meta.get('answer', 'N/A')}")
            
            # Display attachments if any
            attachments_str = meta.get('attachments', '[]')
            try:
                attachments = json.loads(attachments_str)
                if attachments:
                    print(f"\n📎 ATTACHMENTS: {', '.join(attachments)}")
            except:
                pass
            
            print(f"\n" + "─"*100)
        
        print(f"\n✨ Found {len(results['ids'][0])} matching LAQs\n")
    except Exception as e:
        print(f"❌ Search error: {e}")

def chat_laq():
    """Chat with LAQ"""
    query = input("Enter query: ").strip()
    if not query:
        print("❌ Query cannot be empty")
        return
    
    try:
        query_embedding = embed_text(query)
        if not query_embedding:
            print("❌ Could not generate embedding for query")
            return
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["metadatas"]
        )
        
        if not results['ids'] or not results['ids'][0]:
            print("❌ No matching LAQs found")
            return
        
        context = "Relevant LAQs:\n"
        for i, doc_id in enumerate(results['ids'][0]):
            meta = results['metadatas'][0][i]
            attachments_str = meta.get('attachments', '[]')
            try:
                attachments = json.loads(attachments_str)
                attachments_text = f"\nAttachments: {', '.join(attachments)}" if attachments else ""
            except:
                attachments_text = ""
            
            context += f"\nLAQ Type: {meta.get('type', 'N/A')}\nLAQ No: {meta.get('laq_num', 'N/A')}\nMinister: {meta.get('minister', 'N/A')}\nDate: {meta.get('date', 'N/A')}\nQ: {meta.get('question', 'N/A')}\nA: {meta.get('answer', 'N/A')}{attachments_text}\n"
        
        prompt = f"{context}\n\nAnswer this query based on above LAQs:\n{query}"
        response = ollama.generate(model=LLM_MODEL, prompt=prompt, stream=False, options={"temperature": 0})
        
        print("\n" + "="*100)
        print("AI RESPONSE:")
        print("="*100)
        print(response['response'])
        print("="*100)
    except Exception as e:
        print(f"❌ Chat error: {e}")

def clear_db():
    """Clear the database"""
    confirm = input("⚠️ Are you sure you want to clear all data? (yes/no): ").strip().lower()
    if confirm == "yes":
        try:
            client.delete_collection("laqs")
            global collection
            collection = client.get_or_create_collection(name="laqs", metadata={"hnsw:space": "cosine"})
            print("✅ Database cleared successfully")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Operation cancelled")

def main():
    while True:
        print("\n" + "="*100)
        print("LAQ RAG PIPELINE")
        print("="*100)
        print("1. Upload PDF")
        print("2. Search LAQ")
        print("3. Chat with LAQ")
        print("4. Clear Database")
        print("5. Exit")
        print("="*100)
        
        choice = input("Select (1-5): ").strip()
        
        if choice == "1":
            upload_pdf()
        elif choice == "2":
            search_laq()
        elif choice == "3":
            chat_laq()
        elif choice == "4":
            clear_db()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()