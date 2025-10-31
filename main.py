import os
import re
import json
from pathlib import Path
import chromadb
import ollama
from docling.document_converter import DocumentConverter
# from docling.document_converter import ConvertedDocument

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
        return ollama.embed(model="nomic-embed-text", input=text)["embeddings"][0]
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


def structure_laqs_with_mistral(markdown_data, pdf_path):
    """Use Mistral LLM to structure LAQ data from markdown"""
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
2. Treat text like “(a) …”, “(b) …”, “(c) …” as boundaries for new Q&A pairs.
3. Combine lines until a new sub-question or section begins.
4. Keep punctuation and formatting (like “\\n” for line breaks) intact.
5. Output **only valid JSON**. Do not include explanations or extra commentary.

Now extract the structured data in this format from the following text:

{markdown_data[:10000]}
"""


        response = ollama.generate(model="mistral", prompt=prompt, stream=False)
        response_text = response['response'].strip()

        try:
            laq_data = json.loads(response_text)
            return laq_data
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                laq_data = json.loads(json_match.group())
                return laq_data
            else:
                print("❌ Could not parse Mistral response as JSON")
                print(f"Response: {response_text[:200]}")
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
        print(f"  ❓ Q: {qa.get('question', 'N/A')}")
        print(f"  ✅ A: {qa.get('answer', 'N/A')}")
    
    attachments = laq_data.get('attachments', [])
    if attachments:
        print(f"\n📎 Attachments:")
        for att in attachments:
            print(f"  • {att}")
    
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
            print(f"   LAQ #{laq_no} ({laq_type}) | Date: {meta.get('date', 'N/A')} | {match_quality} ({score}%)")
            
            print(f"\n👤 Minister: {meta.get('minister', 'Not mentioned')}")
            
            print(f"\n❓ QUESTION:\n   {meta.get('question', 'N/A')[:200]}..." if len(meta.get('question', '')) > 200 else f"\n❓ QUESTION:\n   {meta.get('question', 'N/A')}")
            
            print(f"\n✅ ANSWER:\n   {meta.get('answer', 'N/A')[:200]}..." if len(meta.get('answer', '')) > 200 else f"\n✅ ANSWER:\n   {meta.get('answer', 'N/A')}")
            
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
        response = ollama.generate(model="mistral", prompt=prompt, stream=False)
        
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