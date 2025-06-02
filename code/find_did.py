import json
from llama_index.core import VectorStoreIndex
from collections import defaultdict
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb

from dotenv import load_dotenv

load_dotenv()
# Connect to your existing ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Adjust path
chroma_collection = chroma_client.get_collection(
    "nber_papers")  # Use your collection name

# Create vector store from existing ChromaDB
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index from existing vector store
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)


# Create retriever with a higher k to catch more matches
# Adjust based on your corpus size
retriever = index.as_retriever(similarity_top_k=100)

# Query for difference-in-difference mentions
nodes = retriever.retrieve("difference-in-difference")

# Extract and organize results
papers_info = defaultdict(list)
unique_papers = set()

for node in nodes:
    # Get paper source/filename
    source = None
    if hasattr(node, 'metadata'):
        source = node.metadata.get('source') or node.metadata.get(
            'file_name') or node.metadata.get('filename')

    if source:
        unique_papers.add(source)
        # Store the text chunk for context
        papers_info[source].append({
            'text': node.text[:200] + "..." if len(node.text) > 200 else node.text,
            'score': getattr(node, 'score', 'N/A')
        })

# Display results
print(
    f"Found {len(unique_papers)} papers mentioning 'difference-in-difference':\n")

for paper in sorted(unique_papers):
    print(f"📄 {paper}")
    # Show first few relevant chunks
    # Show top 3 chunks per paper
    for i, chunk in enumerate(papers_info[paper][:3]):
        print(f"   Chunk {i+1} (Score: {chunk['score']}): {chunk['text']}")
    if len(papers_info[paper]) > 3:
        print(f"   ... and {len(papers_info[paper]) - 3} more chunks")
    print()


# Save to JSON
results_data = {
    'query': 'difference-in-difference',
    'total_papers': len(unique_papers),
    'papers': list(unique_papers),
    'detailed_matches': dict(papers_info)
}

with open('did_papers_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"Results saved to did_papers_results.json")
