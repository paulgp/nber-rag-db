"""
Query interface for the NBER papers RAG database.
Allows users to search and ask questions about the ingested papers.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import chromadb

# Load environment variables
load_dotenv()

def setup_llama_index():
    """Configure LlamaIndex settings."""
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-3.5-turbo")

def load_existing_index():
    """Load the existing vector index from ChromaDB."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("nber_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    return index

def query_papers(query: str, index: VectorStoreIndex, similarity_top_k: int = 5):
    """Query the NBER papers database."""
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
    response = query_engine.query(query)
    return response

def interactive_query():
    """Interactive query interface."""
    print("Setting up RAG system...")
    setup_llama_index()
    
    print("Loading existing index...")
    try:
        index = load_existing_index()
        print("Index loaded successfully!")
    except Exception as e:
        print(f"Error loading index: {e}")
        print("Make sure you've run ingest_pdfs.py first to create the database.")
        return
    
    print("\nNBER Papers RAG Database Ready!")
    print("Ask questions about the working papers. Type 'quit' to exit.\n")
    
    while True:
        query = input("Question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            print("Searching...")
            response = query_papers(query, index)
            print(f"\nAnswer: {response}\n")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing query: {e}")

def single_query(question: str):
    """Query the database with a single question."""
    setup_llama_index()
    index = load_existing_index()
    response = query_papers(question, index)
    return response

if __name__ == "__main__":
    interactive_query()