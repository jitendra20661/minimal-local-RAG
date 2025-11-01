"""Command-line interface for the LAQ RAG system."""

import json
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from config import Config
from database import LAQDatabase, DatabaseError
from embeddings import EmbeddingService, EmbeddingError
from pdf_processor import PDFProcessor, PDFProcessingError, LAQData
from rag import RAGService, RAGError


class CLI:
    """Command-line interface for the LAQ RAG application."""

    def __init__(self, config: Config):
        """Initialize the CLI.

        Args:
            config: Application configuration
        """
        self.config = config
        try:
            self.db = LAQDatabase(config)
            self.embeddings = EmbeddingService(config)
            self.pdf_processor = PDFProcessor(config)
            self.rag = RAGService(config, self.db, self.embeddings)
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise

    def display_menu(self):
        """Display the main menu."""
        print("\n" + "=" * 100)
        print("LAQ RAG PIPELINE")
        print("=" * 100)
        print("1. Upload PDF")
        print("2. Search LAQ")
        print("3. Chat with LAQ")
        print("4. Database Info")
        print("5. Clear Database")
        print("6. Exit")
        print("=" * 100)

    def upload_pdf(self):
        """Handle PDF upload workflow."""
        print("\n" + "=" * 100)
        print("📄 UPLOAD PDF")
        print("=" * 100)

        path = input("Enter PDF path: ").strip()

        try:
            # Process PDF
            laq_data = self.pdf_processor.process_pdf(path)

            # Display extracted data
            self._display_laq_data(laq_data, Path(path).name)

            # Confirm storage
            confirm = input("\n✅ Store this data in database? (yes/no): ").strip().lower()
            if confirm != "yes":
                print("❌ Data not stored")
                return

            # Generate embeddings with progress bar
            print("\n🔄 Generating embeddings...")
            qa_pairs_list = [qa.dict() for qa in laq_data.qa_pairs]

            embeddings_list = []
            with tqdm(total=len(qa_pairs_list), desc="Embedding Q&A pairs", unit="pair") as pbar:
                for qa in qa_pairs_list:
                    text = f"Q: {qa['question']}\nA: {qa['answer']}"
                    try:
                        embedding = self.embeddings.embed_text(text)
                        embeddings_list.append(embedding)
                    except EmbeddingError as e:
                        print(f"\n⚠️ Embedding failed: {e}")
                        embeddings_list.append([])
                    pbar.update(1)

            # Filter out failed embeddings
            valid_qa_pairs = []
            valid_embeddings = []
            for qa, emb in zip(qa_pairs_list, embeddings_list):
                if emb:  # Non-empty embedding
                    valid_qa_pairs.append(qa)
                    valid_embeddings.append(emb)

            if not valid_embeddings:
                print("❌ No valid embeddings generated")
                return

            # Store in database
            print(f"\n💾 Storing in database...")
            laq_dict = laq_data.dict()
            laq_dict["qa_pairs"] = valid_qa_pairs

            stored_count = self.db.store_qa_pairs(
                laq_dict, Path(path).name, valid_embeddings
            )

            print(f"✅ Stored {stored_count}/{len(laq_data.qa_pairs)} Q&A pairs from {Path(path).name}")

        except PDFProcessingError as e:
            print(f"❌ PDF processing error: {e}")
        except DatabaseError as e:
            print(f"❌ Database error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    def search_laq(self):
        """Handle LAQ search workflow."""
        print("\n" + "=" * 100)
        print("🔍 SEARCH LAQ")
        print("=" * 100)

        query = input("Enter query: ").strip()
        if not query:
            print("❌ Query cannot be empty")
            return

        try:
            print("\n🔄 Searching...")
            results = self.rag.search(query)

            if not results:
                print("❌ No results found")
                return

            # Display results
            print("\n" + "=" * 100)
            print(f"🔍 SEARCH RESULTS FOR: {query.upper()}")
            print("=" * 100)
            print(f"Found {len(results)} matching LAQs")

            # Show match quality stats
            stats = self.rag.get_match_quality_stats(results)
            print(f"Match Quality: 🟢 {stats['strong']} strong | 🟡 {stats['moderate']} moderate | 🔴 {stats['weak']} weak")

            for i, result in enumerate(results, 1):
                self._display_search_result(i, result)

        except RAGError as e:
            print(f"❌ Search error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    def chat_laq(self):
        """Handle LAQ chat workflow."""
        print("\n" + "=" * 100)
        print("💬 CHAT WITH LAQ")
        print("=" * 100)

        query = input("Enter query: ").strip()
        if not query:
            print("❌ Query cannot be empty")
            return

        try:
            print("\n🔄 Generating response...")
            response, sources = self.rag.chat(query)

            print("\n" + "=" * 100)
            print("🤖 AI RESPONSE")
            print("=" * 100)
            print(response)
            print("=" * 100)

            # Show sources
            if sources:
                print(f"\n📚 Based on {len(sources)} LAQ(s):")
                for i, source in enumerate(sources, 1):
                    meta = source["metadata"]
                    print(f"  {i}. LAQ #{meta.get('laq_num', 'N/A')} ({source['similarity']}% match)")

        except RAGError as e:
            print(f"❌ Chat error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    def database_info(self):
        """Display database information."""
        print("\n" + "=" * 100)
        print("📊 DATABASE INFO")
        print("=" * 100)

        try:
            count = self.db.get_count()
            print(f"Total documents: {count}")
            print(f"Database path: {self.config.db_path}")
            print(f"Collection name: {self.config.collection_name}")
            print(f"Similarity threshold: {self.config.similarity_threshold}")
        except Exception as e:
            print(f"❌ Error retrieving database info: {e}")

    def clear_database(self):
        """Handle database clearing workflow."""
        print("\n" + "=" * 100)
        print("⚠️  CLEAR DATABASE")
        print("=" * 100)

        confirm = input("⚠️ Are you sure you want to clear all data? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("❌ Operation cancelled")
            return

        try:
            self.db.clear()
            print("✅ Database cleared successfully")
        except DatabaseError as e:
            print(f"❌ Error clearing database: {e}")

    def _display_laq_data(self, laq_data: LAQData, pdf_name: str):
        """Display extracted LAQ data.

        Args:
            laq_data: Extracted LAQ data
            pdf_name: Name of the PDF file
        """
        print("\n" + "=" * 100)
        print("📊 STRUCTURED LAQ DATA (via Mistral)")
        print("=" * 100)
        print(f"📄 PDF: {pdf_name}")
        print(f"📄 Title: {laq_data.pdf_title}")
        print(f"📝 LAQ Type: {laq_data.laq_type}")
        print(f"🔢 LAQ Number: {laq_data.laq_number}")
        print(f"👤 Minister: {laq_data.minister}")
        print(f"📅 Date: {laq_data.date}")
        if laq_data.tabled_by:
            print(f"📋 Tabled by: {laq_data.tabled_by}")

        print(f"\n❓ Question-Answer Pairs: {len(laq_data.qa_pairs)}")
        print("─" * 100)

        for idx, qa in enumerate(laq_data.qa_pairs, 1):
            print(f"\n[Q&A Pair {idx}]")
            print(f"  ❓ Q: {qa.question[:200]}..." if len(qa.question) > 200 else f"  ❓ Q: {qa.question}")
            print(f"  ✅ A: {qa.answer[:200]}..." if len(qa.answer) > 200 else f"  ✅ A: {qa.answer}")

        if laq_data.attachments:
            print(f"\n📎 Attachments:")
            for att in laq_data.attachments:
                print(f"  • {att}")

        print("=" * 100)

    def _display_search_result(self, index: int, result: dict):
        """Display a single search result.

        Args:
            index: Result index (1-based)
            result: Search result dictionary
        """
        meta = result["metadata"]
        similarity = result["similarity"]
        match_color = result["match_color"]
        match_quality = result["match_quality"]

        pdf_name = meta.get("pdf", "Unknown").replace(".pdf", "").upper()
        laq_type = meta.get("type", "N/A")
        laq_no = meta.get("laq_num", "N/A")

        print(f"\n┌{'─' * 98}┐")
        print(f"│ RESULT #{index} {' ' * 85}│")
        print(f"└{'─' * 98}┘")

        print(f"\n📍 SOURCE: {pdf_name}")
        print(f"   LAQ #{laq_no} ({laq_type}) | Date: {meta.get('date', 'N/A')} | {match_color} {match_quality} ({similarity}%)")

        print(f"\n👤 Minister: {meta.get('minister', 'Not mentioned')}")

        question = meta.get("question", "N/A")
        print(f"\n❓ QUESTION:")
        print(f"   {question[:300]}..." if len(question) > 300 else f"   {question}")

        answer = meta.get("answer", "N/A")
        print(f"\n✅ ANSWER:")
        print(f"   {answer[:300]}..." if len(answer) > 300 else f"   {answer}")

        # Display attachments if any
        try:
            attachments = json.loads(meta.get("attachments", "[]"))
            if attachments:
                print(f"\n📎 ATTACHMENTS: {', '.join(attachments)}")
        except:
            pass

        print(f"\n{'─' * 100}")

    def run(self):
        """Run the CLI application."""
        print("\n" + "=" * 100)
        print("🚀 WELCOME TO LAQ RAG SYSTEM")
        print("=" * 100)
        print(f"Database: {self.db.get_count()} documents loaded")

        while True:
            try:
                self.display_menu()
                choice = input("Select (1-6): ").strip()

                if choice == "1":
                    self.upload_pdf()
                elif choice == "2":
                    self.search_laq()
                elif choice == "3":
                    self.chat_laq()
                elif choice == "4":
                    self.database_info()
                elif choice == "5":
                    self.clear_database()
                elif choice == "6":
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-6.")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
