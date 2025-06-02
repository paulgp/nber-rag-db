"""
PDF ingestion script for NBER working papers.
Loads PDFs from data/pdf/ directory and creates a vector index.
"""

import os
import json
import hashlib
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

def load_ingested_files():
    """Load the set of already ingested files from tracking file."""
    tracking_file = Path("./ingested_files.json")
    if tracking_file.exists():
        with open(tracking_file, 'r') as f:
            return set(json.load(f))
    return set()

def save_ingested_files(ingested_files):
    """Save the set of ingested files to tracking file."""
    tracking_file = Path("./ingested_files.json")
    with open(tracking_file, 'w') as f:
        json.dump(list(ingested_files), f, indent=2)

def get_file_hash(file_path):
    """Generate a hash for a file based on its path and modification time."""
    stat = file_path.stat()
    content = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()

def filter_new_documents(documents, ingested_files):
    """Filter out documents that have already been ingested."""
    new_documents = []
    new_hashes = set()
    
    for doc in documents:
        if hasattr(doc, 'metadata') and 'file_path' in doc.metadata:
            file_path = Path(doc.metadata['file_path'])
            file_hash = get_file_hash(file_path)
            
            if file_hash not in ingested_files:
                new_documents.append(doc)
                new_hashes.add(file_hash)
                print(f"New document: {file_path.name}")
            else:
                print(f"Skipping already ingested: {file_path.name}")
        else:
            new_documents.append(doc)
    
    return new_documents, new_hashes


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

def create_index_in_blocks(documents, vector_store, block_size=5):
    """Create vector index from documents in blocks."""
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Try to load existing index or create new one
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        print("Loaded existing index")
    except:
        # Create new index if none exists
        if documents:
            print("Creating new index with first block...")
            first_block = documents[:min(block_size, len(documents))]
            index = VectorStoreIndex.from_documents(
                first_block,
                storage_context=storage_context,
                show_progress=True
            )
            documents = documents[len(first_block):]
        else:
            print("No documents to create index")
            return None
    
    # Process remaining documents in blocks
    total_docs = len(documents)
    for i in range(0, total_docs, block_size):
        block_end = min(i + block_size, total_docs)
        block = documents[i:block_end]
        
        print(f"\nProcessing block {i//block_size + 1}: documents {i+1}-{block_end} of {total_docs}")
        for j, doc in enumerate(block, 1):
            if hasattr(doc, 'metadata') and 'file_path' in doc.metadata:
                file_name = Path(doc.metadata['file_path']).name
                print(f"  [{i+j}/{total_docs}] Processing: {file_name}")
            else:
                print(f"  [{i+j}/{total_docs}] Processing document {i+j}")
        
        # Add documents to existing index
        for doc in block:
            index.insert(doc)
        
        print(f"Completed block {i//block_size + 1}")
    
    return index

def process_documents_with_progress(documents, document_hashes, ingested_files, vector_store, block_size=5):
    """Process documents in blocks with progress tracking and resumability."""
    processed_count = 0
    
    try:
        print(f"\n🚀 Starting to process {len(documents)} new documents in blocks of {block_size}")
        print("💡 You can safely quit (Ctrl+C) and resume later - progress will be saved!\n")
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Try to load existing index or create new one
        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            print("Loaded existing index")
        except:
            # Create new index if none exists
            if documents:
                print("Creating new index...")
                index = VectorStoreIndex([], storage_context=storage_context)
            else:
                print("No documents to create index")
                return None, 0
        
        # Process documents in blocks
        total_docs = len(documents)
        doc_hash_pairs = list(zip(documents, document_hashes))
        
        for i in range(0, total_docs, block_size):
            block_end = min(i + block_size, total_docs)
            block_pairs = doc_hash_pairs[i:block_end]
            
            print(f"\n📦 Processing block {i//block_size + 1}: documents {i+1}-{block_end} of {total_docs}")
            
            # Process each document in the block
            for j, (doc, doc_hash) in enumerate(block_pairs, 1):
                if hasattr(doc, 'metadata') and 'file_path' in doc.metadata:
                    file_name = Path(doc.metadata['file_path']).name
                    print(f"  📄 [{i+j}/{total_docs}] Processing: {file_name}")
                else:
                    print(f"  📄 [{i+j}/{total_docs}] Processing document {i+j}")
                
                # Add document to index
                index.insert(doc)
                
                # Mark this specific document as processed
                ingested_files.add(doc_hash)
                save_ingested_files(ingested_files)
                processed_count += 1
                
                print(f"     ✅ Completed and saved progress")
            
            print(f"✅ Completed block {i//block_size + 1} ({block_end - i} documents)")
        
        return index, processed_count
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Process interrupted by user")
        print(f"📊 Progress: {processed_count}/{len(documents)} documents processed")
        print("💾 Progress has been saved - you can resume by running the script again")
        return None, processed_count

def main():
    """Main ingestion function."""
    print("Setting up LlamaIndex...")
    setup_llama_index()
    
    print("Creating vector store...")
    vector_store = create_vector_store()
    
    print("Loading ingested files tracking...")
    ingested_files = load_ingested_files()
    print(f"Found {len(ingested_files)} previously ingested files")
    
    print("Loading documents...")
    documents = load_documents("/Users/psg24/repos/update-words/Data/PDF/NBER")
    
    print("Filtering new documents...")
    new_documents, new_hashes = filter_new_documents(documents, ingested_files)
    
    if not new_documents:
        print("✅ No new documents to ingest! All files are up to date.")
        return
    
    # Process documents with resumability
    index, processed_count = process_documents_with_progress(
        new_documents, new_hashes, ingested_files, vector_store, block_size=3
    )
    
    if index:
        print("\n🎉 Ingestion complete!")
        print(f"✅ Added {processed_count} new documents to index")
        print(f"📁 Total tracked files: {len(ingested_files)}")
        print("💾 Vector store saved to ./chroma_db")
    else:
        print(f"\n⚠️  Ingestion interrupted after processing {processed_count} documents")
        print("🔄 Run the script again to continue processing remaining files")

if __name__ == "__main__":
    main()