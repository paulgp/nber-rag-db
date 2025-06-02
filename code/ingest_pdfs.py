"""
PDF ingestion script for NBER working papers.
Loads PDFs from data/pdf/ directory and creates a vector index.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
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

def create_vector_store():
    """Create and return a ChromaDB vector store."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("nber_papers")
    return ChromaVectorStore(chroma_collection=chroma_collection)

def load_documents(data_dir: str = "../data/pdf"):
    """Load PDF documents from the specified directory."""
    pdf_dir = Path(__file__).parent / data_dir
    
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    
    print(f"Loading PDFs from: {pdf_dir}")
    reader = SimpleDirectoryReader(
        input_dir=str(pdf_dir)
    )
    
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents")
    return documents

def create_index(documents, vector_store):
    """Create vector index from documents."""
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )
    return index

def main():
    """Main ingestion function."""
    print("Setting up LlamaIndex...")
    setup_llama_index()
    
    print("Creating vector store...")
    vector_store = create_vector_store()
    
    print("Loading documents...")
    documents = load_documents()
    
    print("Creating vector index...")
    index = create_index(documents, vector_store)
    
    print("Ingestion complete!")
    print(f"Index created with {len(documents)} documents")
    print("Vector store saved to ./chroma_db")

if __name__ == "__main__":
    main()