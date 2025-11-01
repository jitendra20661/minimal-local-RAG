"""
LAQ RAG System - Main Entry Point

A minimal, privacy-friendly Retrieval-Augmented Generation (RAG) system
for Legislative Assembly Questions (LAQs) that runs entirely locally.
"""

import sys

from config import Config
from cli import CLI
from embeddings import OllamaConnectionError, OllamaModelNotFoundError


def main():
    """Main entry point for the LAQ RAG application."""
    try:
        # Load configuration
        config = Config()

        # Initialize and run CLI
        cli = CLI(config)
        cli.run()

    except OllamaConnectionError as e:
        print(f"\n❌ Ollama Connection Error:\n{e}\n")
        print("Please ensure Ollama is installed and running:")
        print("  1. Install: https://ollama.ai")
        print("  2. Start: ollama serve")
        print("  3. Pull models: ollama pull mistral && ollama pull nomic-embed-text")
        sys.exit(1)

    except OllamaModelNotFoundError as e:
        print(f"\n❌ Model Not Found:\n{e}\n")
        sys.exit(1)

    except ValueError as e:
        print(f"\n❌ Configuration Error:\n{e}\n")
        print("Please check your .env file or environment variables.")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
