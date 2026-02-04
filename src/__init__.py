"""
Multimodal RAG for Enterprise Document Analysis

A proof-of-concept RAG pipeline for analyzing enterprise documents
including PDFs, tables, and images.
"""

from .models import ContentType, DocumentChunk, RetrievedChunk, AnalysisOutput
from .ingestion import DocumentIngester
from .chunking import SmartChunker
from .indexing import HybridRAGIndex
from .agent import RAGAgent
from .evaluation import RAGEvaluator

__version__ = "0.1.0"
__all__ = [
    "ContentType",
    "DocumentChunk", 
    "RetrievedChunk",
    "AnalysisOutput",
    "DocumentIngester",
    "SmartChunker",
    "HybridRAGIndex",
    "RAGAgent",
    "RAGEvaluator",
]
