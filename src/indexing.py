"""
Hybrid RAG index combining dense and sparse retrieval.

Features:
- Dense retrieval via sentence-transformers + ChromaDB
- Sparse retrieval via BM25
- Reciprocal Rank Fusion for combining results
- Explainable retrieval with method attribution
"""

import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from loguru import logger
from tqdm.auto import tqdm

from .models import DocumentChunk, RetrievedChunk, ContentType


class HybridRAGIndex:
    """
    Hybrid RAG index combining dense and sparse retrieval methods.
    
    Architecture:
    - Dense Index: ChromaDB with sentence-transformer embeddings
    - Sparse Index: BM25 for keyword matching
    - Fusion: Reciprocal Rank Fusion (RRF)
    
    The hybrid approach captures both semantic similarity (dense)
    and exact keyword matches (sparse), providing robust retrieval
    across different query types.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "documents",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the hybrid RAG index.
        
        Args:
            embedding_model: Sentence transformer model name
            collection_name: Name for ChromaDB collection
            persist_directory: Directory to persist ChromaDB (None for in-memory)
        """
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.model_name = embedding_model
        
        # Initialize ChromaDB
        if persist_directory:
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.chroma_client = chromadb.Client()
        
        # Create embedding function for ChromaDB
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )
        
        # BM25 index components
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []
        self.chunks: List[DocumentChunk] = []
        self.chunk_id_to_index: Dict[str, int] = {}
        
        logger.info(f"HybridRAGIndex initialized with collection: {collection_name}")
    
    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100):
        """
        Add chunks to both dense and sparse indices.
        
        Args:
            chunks: List of DocumentChunk objects to index
            batch_size: Batch size for indexing operations
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Indexing {len(chunks)} chunks...")
        
        # Prepare data for ChromaDB
        all_ids = []
        all_documents = []
        all_metadatas = []
        
        for chunk in tqdm(chunks, desc="Preparing chunks"):
            chunk_id = chunk.chunk_id
            all_ids.append(chunk_id)
            all_documents.append(chunk.content)
            all_metadatas.append({
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "content_type": chunk.content_type.value,
                "parent_section": chunk.parent_section or "",
                "chunk_index": chunk.chunk_index
            })
            
            # Track for BM25
            self.chunk_id_to_index[chunk_id] = len(self.chunks)
            self.chunks.append(chunk)
            self.bm25_corpus.append(self._tokenize(chunk.content))
        
        # Add to ChromaDB in batches
        for i in tqdm(range(0, len(all_ids), batch_size), desc="Indexing to ChromaDB"):
            batch_end = min(i + batch_size, len(all_ids))
            self.collection.add(
                ids=all_ids[i:batch_end],
                documents=all_documents[i:batch_end],
                metadatas=all_metadatas[i:batch_end]
            )
        
        # Rebuild BM25 index
        logger.info("Building BM25 index...")
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        logger.info(f"Successfully indexed {len(chunks)} chunks")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        # Lowercase and extract alphanumeric tokens
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def search_dense(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Semantic search using dense embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        where_filter = None
        if filter_metadata:
            where_filter = filter_metadata
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved = []
        if results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                if chunk_id in self.chunk_id_to_index:
                    chunk = self.chunks[self.chunk_id_to_index[chunk_id]]
                    # Convert distance to similarity (for cosine distance)
                    distance = results['distances'][0][i]
                    similarity = 1 - distance
                    retrieved.append((chunk, similarity))
        
        return retrieved
    
    def search_sparse(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Keyword search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk, bm25_score) tuples
        """
        if not self.bm25_index:
            logger.warning("BM25 index not built")
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieved = []
        for idx in top_indices:
            if scores[idx] > 0:
                retrieved.append((self.chunks[idx], float(scores[idx])))
        
        return retrieved
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60
    ) -> List[RetrievedChunk]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).
        
        RRF combines rankings from different retrieval methods using:
        score = sum(1 / (k + rank)) for each method
        
        Args:
            query: Search query
            top_k: Final number of results to return
            dense_weight: Weight for dense retrieval in RRF
            sparse_weight: Weight for sparse retrieval in RRF
            rrf_k: RRF constant (typically 60)
            
        Returns:
            List of RetrievedChunk objects with scores and explanations
        """
        # Get results from both methods (fetch more than needed for fusion)
        fetch_k = top_k * 3
        dense_results = self.search_dense(query, top_k=fetch_k)
        sparse_results = self.search_sparse(query, top_k=fetch_k)
        
        # Calculate RRF scores
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, DocumentChunk] = {}
        method_map: Dict[str, List[str]] = defaultdict(list)
        score_details: Dict[str, Dict] = {}
        
        # Process dense results
        for rank, (chunk, score) in enumerate(dense_results):
            chunk_id = chunk.chunk_id
            rrf_scores[chunk_id] += dense_weight * (1 / (rrf_k + rank + 1))
            chunk_map[chunk_id] = chunk
            method_map[chunk_id].append('dense')
            
            if chunk_id not in score_details:
                score_details[chunk_id] = {}
            score_details[chunk_id]['dense_score'] = score
            score_details[chunk_id]['dense_rank'] = rank + 1
        
        # Process sparse results
        for rank, (chunk, score) in enumerate(sparse_results):
            chunk_id = chunk.chunk_id
            rrf_scores[chunk_id] += sparse_weight * (1 / (rrf_k + rank + 1))
            chunk_map[chunk_id] = chunk
            method_map[chunk_id].append('sparse')
            
            if chunk_id not in score_details:
                score_details[chunk_id] = {}
            score_details[chunk_id]['sparse_score'] = score
            score_details[chunk_id]['sparse_rank'] = rank + 1
        
        # Sort by RRF score and take top-k
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Build retrieved chunks with explanations
        retrieved = []
        for chunk_id, rrf_score in sorted_chunks:
            chunk = chunk_map[chunk_id]
            methods = method_map[chunk_id]
            details = score_details[chunk_id]
            
            # Generate explanation
            explanation = self._generate_retrieval_explanation(
                chunk, methods, details, query
            )
            
            # Determine retrieval method label
            method_str = 'hybrid' if len(methods) > 1 else methods[0]
            
            retrieved.append(RetrievedChunk(
                chunk=chunk,
                score=rrf_score,
                retrieval_method=method_str,
                explanation=explanation
            ))
        
        return retrieved
    
    def _generate_retrieval_explanation(
        self,
        chunk: DocumentChunk,
        methods: List[str],
        details: Dict,
        query: str
    ) -> str:
        """
        Generate human-readable explanation for why chunk was retrieved.
        
        This provides transparency into the retrieval process, helping
        users understand and trust the results.
        """
        parts = []
        
        if 'dense' in methods and 'sparse' in methods:
            parts.append(
                f"Retrieved by BOTH semantic similarity "
                f"(rank #{details.get('dense_rank', '?')}, "
                f"score: {details.get('dense_score', 0):.3f}) "
                f"and keyword matching "
                f"(rank #{details.get('sparse_rank', '?')}, "
                f"BM25: {details.get('sparse_score', 0):.2f})."
            )
        elif 'dense' in methods:
            parts.append(
                f"Retrieved by semantic similarity "
                f"(rank #{details.get('dense_rank', '?')}, "
                f"cosine similarity: {details.get('dense_score', 0):.3f})."
            )
        else:
            parts.append(
                f"Retrieved by keyword matching "
                f"(rank #{details.get('sparse_rank', '?')}, "
                f"BM25 score: {details.get('sparse_score', 0):.2f})."
            )
        
        # Add content type context
        if chunk.content_type == ContentType.TABLE:
            parts.append("Contains structured tabular data.")
        elif chunk.content_type == ContentType.IMAGE:
            parts.append("Contains image reference.")
        
        # Add source context
        parts.append(f"Source: {chunk.document_name}, page {chunk.page_number}.")
        
        if chunk.parent_section:
            parts.append(f"Section: {chunk.parent_section}")
        
        return " ".join(parts)
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        content_type_counts = defaultdict(int)
        document_counts = defaultdict(int)
        
        for chunk in self.chunks:
            content_type_counts[chunk.content_type.value] += 1
            document_counts[chunk.document_name] += 1
        
        return {
            "total_chunks": len(self.chunks),
            "documents": len(document_counts),
            "document_details": dict(document_counts),
            "content_types": dict(content_type_counts),
            "embedding_model": self.model_name,
            "bm25_indexed": self.bm25_index is not None
        }
    
    def clear(self):
        """Clear all indexed data."""
        # Clear ChromaDB collection
        self.chroma_client.delete_collection(self.collection.name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Clear BM25 data
        self.bm25_index = None
        self.bm25_corpus = []
        self.chunks = []
        self.chunk_id_to_index = {}
        
        logger.info("Index cleared")
