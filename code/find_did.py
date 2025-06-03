import json
import argparse
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


def find_papers_with_similarity_cutoff(query, similarity_cutoff=0.0, max_results=100):
    """Find papers with similarity score above the cutoff."""
    # Create retriever with a higher k to catch more matches
    retriever = index.as_retriever(similarity_top_k=max_results)
    
    # Query for matches
    nodes = retriever.retrieve(query)
    
    return nodes, query


def process_results(nodes, query, similarity_cutoff=0.0):
    """Process and filter results based on similarity cutoff."""
    papers_info = defaultdict(list)
    unique_papers = set()
    total_chunks = 0
    filtered_chunks = 0
    
    for node in nodes:
        total_chunks += 1
        
        # Check similarity score
        score = getattr(node, 'score', None)
        if score is not None and score < similarity_cutoff:
            continue
        
        filtered_chunks += 1
        
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
                'score': score if score is not None else 'N/A'
            })
    
    return papers_info, unique_papers, total_chunks, filtered_chunks


def display_and_save_results(papers_info, unique_papers, query, similarity_cutoff, total_chunks, filtered_chunks):
    """Display and save filtered results."""
    # Display results
    print(f"Query: '{query}'")
    print(f"Similarity cutoff: {similarity_cutoff}")
    print(f"Total chunks retrieved: {total_chunks}")
    print(f"Chunks above cutoff: {filtered_chunks}")
    print(f"Found {len(unique_papers)} papers with similarity >= {similarity_cutoff}:\n")

    for paper in sorted(unique_papers):
        print(f"📄 {paper}")
        # Show first few relevant chunks
        # Sort chunks by score (highest first) and show top 3
        sorted_chunks = sorted(papers_info[paper], 
                             key=lambda x: x['score'] if isinstance(x['score'], (int, float)) else 0, 
                             reverse=True)
        
        for i, chunk in enumerate(sorted_chunks[:3]):
            score_str = f"{chunk['score']:.4f}" if isinstance(chunk['score'], (int, float)) else str(chunk['score'])
            #print(f"   Chunk {i+1} (Score: {score_str}): {chunk['text']}")
        # if len(papers_info[paper]) > 3:
        #     print(f"   ... and {len(papers_info[paper]) - 3} more chunks")
        # print()

    # Save to JSON
    results_data = {
        'query': query,
        'similarity_cutoff': similarity_cutoff,
        'total_chunks_retrieved': total_chunks,
        'chunks_above_cutoff': filtered_chunks,
        'total_papers': len(unique_papers),
        'papers': list(unique_papers),
        'detailed_matches': dict(papers_info)
    }

    filename = f"{query.replace(' ', '_').replace('/', '_')}_cutoff_{similarity_cutoff}_results.json"
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"Results saved to {filename}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Find papers with similarity score above cutoff')
    parser.add_argument('--query', '-q', default='financial event study', 
                       help='Search query (default: "financial event study")')
    parser.add_argument('--cutoff', '-c', type=float, default=0.0,
                       help='Similarity score cutoff (default: 0.0)')
    parser.add_argument('--max-results', '-m', type=int, default=100,
                       help='Maximum number of results to retrieve (default: 100)')
    
    args = parser.parse_args()
    
    # Find papers
    nodes, query = find_papers_with_similarity_cutoff(args.query, args.cutoff, args.max_results)
    
    # Process and filter results
    papers_info, unique_papers, total_chunks, filtered_chunks = process_results(nodes, query, args.cutoff)
    
    # Display and save results
    display_and_save_results(papers_info, unique_papers, query, args.cutoff, total_chunks, filtered_chunks)


if __name__ == "__main__":
    main()