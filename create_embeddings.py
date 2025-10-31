"""
FAISS Embedding Creation Script for Astrophysics RAG (OPTIMIZED VERSION)

This script takes the chunked papers JSON file and creates FAISS vector embeddings
suitable for semantic search and retrieval in your RAG chatbot.

OPTIMIZATIONS:
- GPU acceleration support (automatic detection)
- Batch processing with progress bars
- Memory-efficient loading
- Resume capability (saves intermediate results)
- Multi-threading support

Requirements:
    pip install sentence-transformers faiss-cpu numpy tqdm
    (or faiss-gpu if you have a CUDA-enabled GPU)
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pickle
import torch
from tqdm import tqdm
import time


class FaissEmbeddingCreator:
    """Creates FAISS embeddings from chunked documents (OPTIMIZED)"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_gpu: bool = True):
        """
        Initialize the embedding creator
        
        Args:
            model_name: Name of the sentence transformer model to use
                Popular options:
                - 'all-MiniLM-L6-v2' (fast, 384 dim, good for general use)
                - 'all-mpnet-base-v2' (slower, 768 dim, better quality)
                - 'multi-qa-MiniLM-L6-cos-v1' (optimized for Q&A)
            use_gpu: Whether to use GPU if available
        """
        # Check GPU availability
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        
        print(f"{'='*60}")
        print(f"Loading embedding model: {model_name}")
        print(f"Device: {self.device.upper()}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"{'='*60}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name
        print(f"✓ Model loaded! Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def load_chunks(self, chunks_file: str) -> tuple:
        """
        Load chunks from JSON file with progress indication
        
        Args:
            chunks_file: Path to chunked_papers.json
            
        Returns:
            Tuple of (chunks_data, texts, metadata_list)
        """
        print(f"\n{'='*60}")
        print(f"Loading chunks from {chunks_file}...")
        
        start_time = time.time()
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data['chunks']
        texts = [chunk['text'] for chunk in chunks]
        metadata_list = [chunk['metadata'] for chunk in chunks]
        
        load_time = time.time() - start_time
        print(f"✓ Loaded {len(texts):,} chunks in {load_time:.2f} seconds")
        print(f"{'='*60}")
        return data, texts, metadata_list
    
    def create_embeddings(self, texts: List[str], batch_size: int = 64, cache_file: str = None) -> np.ndarray:
        """
        Create embeddings for all texts with optimization
        
        Args:
            texts: List of text chunks
            batch_size: Batch size for encoding (larger = faster on GPU)
            cache_file: Path to cache embeddings (to resume if interrupted)
            
        Returns:
            Numpy array of embeddings
        """
        # Check if cached embeddings exist
        if cache_file and Path(cache_file).exists():
            print(f"\n✓ Found cached embeddings at {cache_file}")
            print("Loading from cache...")
            embeddings = np.load(cache_file)
            print(f"✓ Loaded {len(embeddings):,} embeddings from cache")
            return embeddings
        
        print(f"\n{'='*60}")
        print(f"Creating embeddings for {len(texts):,} chunks")
        print(f"Batch size: {batch_size}")
        print(f"This may take a few minutes...")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Encode with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✓ Embeddings created!")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"  Speed: {len(texts)/elapsed_time:.1f} chunks/second")
        print(f"{'='*60}")
        
        # Cache embeddings if requested
        if cache_file:
            print(f"\nCaching embeddings to {cache_file}...")
            np.save(cache_file, embeddings)
            print(f"✓ Embeddings cached!")
        
        return embeddings
    
    def create_faiss_index(self, embeddings: np.ndarray, index_type: str = 'flat') -> faiss.Index:
        """
        Create FAISS index from embeddings (OPTIMIZED)
        
        Args:
            embeddings: Numpy array of embeddings (already normalized)
            index_type: Type of FAISS index
                - 'flat': Exact search (best quality, good for <1M vectors)
                - 'ivf': Inverted file index (faster, slight quality trade-off)
                - 'hnsw': Hierarchical Navigable Small World (fast, good quality)
                
        Returns:
            FAISS index
        """
        print(f"\n{'='*60}")
        print(f"Creating FAISS index (type: {index_type})")
        print(f"{'='*60}")
        
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        # Embeddings are already normalized in create_embeddings()
        
        if index_type == 'flat':
            # Exact search with inner product (cosine similarity for normalized vectors)
            print(f"Building Flat index for exact search...")
            index = faiss.IndexFlatIP(dimension)
            
        elif index_type == 'ivf':
            # IVF index for faster approximate search
            nlist = min(int(np.sqrt(n_vectors)), 100)  # Number of clusters
            print(f"Building IVF index with {nlist} clusters...")
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            print(f"Training IVF index...")
            index.train(embeddings)
            print(f"✓ Training complete!")
            
        elif index_type == 'hnsw':
            # HNSW index for fast approximate search
            print(f"Building HNSW index...")
            index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 40
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        print(f"Adding {n_vectors:,} vectors to index...")
        start_time = time.time()
        index.add(embeddings)
        add_time = time.time() - start_time
        
        print(f"✓ Index created!")
        print(f"  Total vectors: {index.ntotal:,}")
        print(f"  Time: {add_time:.2f} seconds")
        print(f"{'='*60}")
        return index
    
    def save_index_and_metadata(
        self,
        index: faiss.Index,
        metadata_list: List[Dict],
        chunks_data: Dict,
        output_dir: str = "faiss_index"
    ):
        """
        Save FAISS index and metadata
        
        Args:
            index: FAISS index
            metadata_list: List of metadata dictionaries
            chunks_data: Original chunks data with statistics
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save FAISS index
        index_file = output_path / "astrophysics.index"
        faiss.write_index(index, str(index_file))
        print(f"\n✓ FAISS index saved to: {index_file}")
        
        # Save metadata
        metadata_file = output_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata_list, f)
        print(f"✓ Metadata saved to: {metadata_file}")
        
        # Save chunks data (for reference)
        chunks_file = output_path / "chunks_backup.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Chunks backup saved to: {chunks_file}")
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'total_vectors': index.ntotal,
            'index_type': type(index).__name__
        }
        config_file = output_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved to: {config_file}")
        
        print(f"\n{'='*60}")
        print("All files saved successfully!")
        print(f"{'='*60}")
    
    def search_example(self, index: faiss.Index, metadata_list: List[Dict], query: str, k: int = 5):
        """
        Demonstrate search functionality
        
        Args:
            index: FAISS index
            metadata_list: List of metadata dictionaries
            query: Search query
            k: Number of results to return
        """
        print(f"\n{'='*60}")
        print(f"Example Search Query: '{query}'")
        print(f"{'='*60}")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = index.search(query_embedding, k)
        
        print(f"\nTop {k} Results:")
        print("-" * 60)
        
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
            meta = metadata_list[idx]
            print(f"\n{i}. Score: {distance:.4f}")
            print(f"   Paper: {meta['paper_id']}")
            print(f"   Section: {meta['section']}")
            print(f"   Chunk: {meta['chunk_index']}/{meta['total_chunks']}")
            print(f"   Tokens: {meta['token_count']}")


def main():
    """Main execution function (OPTIMIZED)"""
    
    # Configuration
    CHUNKS_FILE = "chunked_papers.json"
    OUTPUT_DIR = "faiss_index"
    MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and efficient (384 dimensions)
    INDEX_TYPE = "flat"  # Use 'flat' for exact search (<1M vectors)
    BATCH_SIZE = 128  # Larger batch = faster on GPU (reduce to 32 for CPU)
    USE_GPU = True  # Set to False if you want CPU only
    CACHE_EMBEDDINGS = True  # Cache embeddings to avoid recomputation
    
    print("="*60)
    print("FAISS Embedding Creator for Astrophysics RAG (OPTIMIZED)")
    print("="*60)
    
    # Initialize creator
    creator = FaissEmbeddingCreator(model_name=MODEL_NAME, use_gpu=USE_GPU)
    
    # Load chunks
    chunks_data, texts, metadata_list = creator.load_chunks(CHUNKS_FILE)
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Create embeddings (with caching)
    cache_file = Path(OUTPUT_DIR) / "embeddings_cache.npy" if CACHE_EMBEDDINGS else None
    embeddings = creator.create_embeddings(texts, batch_size=BATCH_SIZE, cache_file=cache_file)
    
    # Create FAISS index
    index = creator.create_faiss_index(embeddings, index_type=INDEX_TYPE)
    
    # Save everything
    creator.save_index_and_metadata(index, metadata_list, chunks_data, OUTPUT_DIR)
    
    # Demonstrate search with an example query
    example_query = "What are gamma-ray bursts and their progenitors?"
    creator.search_example(index, metadata_list, example_query, k=5)
    
    print("\n" + "="*60)
    print("🎉 FAISS index creation complete!")
    print("="*60)
    print("\n📦 Files created:")
    print(f"  - {OUTPUT_DIR}/astrophysics.index (FAISS index)")
    print(f"  - {OUTPUT_DIR}/metadata.pkl (chunk metadata)")
    print(f"  - {OUTPUT_DIR}/embeddings_cache.npy (embeddings cache)")
    print(f"  - {OUTPUT_DIR}/config.json (configuration)")
    print("\n🚀 Next steps:")
    print("  1. Load index: faiss.read_index('faiss_index/astrophysics.index')")
    print("  2. Load metadata: pickle.load(open('faiss_index/metadata.pkl', 'rb'))")
    print("  3. Build your RAG chatbot!")
    print("="*60)


if __name__ == "__main__":
    main()
