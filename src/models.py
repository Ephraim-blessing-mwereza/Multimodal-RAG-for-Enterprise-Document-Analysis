"""
Data models for the Multimodal RAG system.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional

import pandas as pd


class ContentType(Enum):
    """Types of content that can be extracted from documents."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    METADATA = "metadata"


@dataclass
class DocumentChunk:
    """
    Represents a chunk of content from a document.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        content: The text content of the chunk
        content_type: Type of content (text, table, image)
        document_name: Source document name
        page_number: Page number in source document
        metadata: Additional metadata
        table_data: DataFrame for table chunks
        table_summary: Natural language summary of table
        image_path: Path to extracted image
        image_description: Description of image content
        start_char: Starting character position in document
        end_char: Ending character position in document
        parent_section: Parent section header
        chunk_index: Index of chunk in document
    """
    chunk_id: str
    content: str
    content_type: ContentType
    document_name: str
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # For tables
    table_data: Optional[pd.DataFrame] = None
    table_summary: Optional[str] = None
    
    # For images
    image_path: Optional[str] = None
    image_description: Optional[str] = None
    
    # Position in document
    start_char: int = 0
    end_char: int = 0
    
    # Hierarchy
    parent_section: Optional[str] = None
    chunk_index: int = 0
    
    def __post_init__(self):
        """Generate unique ID if not provided."""
        if not self.chunk_id:
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.chunk_id = f"{self.document_name}_{self.page_number}_{content_hash}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "content_type": self.content_type.value,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "metadata": self.metadata,
            "table_summary": self.table_summary,
            "image_description": self.image_description,
            "parent_section": self.parent_section,
        }
    
    def __repr__(self) -> str:
        return f"DocumentChunk(id={self.chunk_id}, type={self.content_type.value}, doc={self.document_name}, page={self.page_number})"


@dataclass
class RetrievedChunk:
    """
    A chunk with retrieval information.
    
    Attributes:
        chunk: The retrieved document chunk
        score: Relevance score (higher is better)
        retrieval_method: Method used ('dense', 'sparse', 'hybrid')
        explanation: Human-readable explanation of why retrieved
    """
    chunk: DocumentChunk
    score: float
    retrieval_method: str
    explanation: str = ""
    
    def to_citation(self) -> Dict:
        """Generate citation format."""
        return {
            "document": self.chunk.document_name,
            "page": self.chunk.page_number,
            "content_preview": (
                self.chunk.content[:200] + "..." 
                if len(self.chunk.content) > 200 
                else self.chunk.content
            ),
            "relevance_score": round(self.score, 3),
            "retrieval_method": self.retrieval_method,
            "explanation": self.explanation
        }


@dataclass
class KeyFinding:
    """A key finding extracted from documents."""
    finding: str
    category: str = ""
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)


@dataclass
class NumericalData:
    """Extracted numerical data point."""
    metric_name: str
    value: Any
    unit: str = ""
    context: str = ""
    source_location: str = ""
    verified: bool = False


@dataclass
class RiskFlag:
    """A risk flag identified in documents."""
    risk_type: str
    description: str
    severity: str = "medium"  # low, medium, high, critical
    mitigation: str = ""
    evidence: List[str] = field(default_factory=list)


@dataclass
class AnalysisOutput:
    """
    Structured output from the RAG system.
    
    Attributes:
        query: Original user query
        summary: Natural language summary/answer
        key_findings: List of key findings with sources
        extracted_data: Numerical data extracted
        risk_flags: Identified risks and concerns
        citations: Sources used with relevance scores
        metadata: Processing metadata
    """
    query: str
    summary: str
    key_findings: List[Dict[str, Any]]
    extracted_data: List[Dict[str, Any]]
    risk_flags: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(asdict(self), indent=indent, default=str)
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return asdict(self)
    
    def save(self, filepath: str) -> None:
        """Save output to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> "AnalysisOutput":
        """Load output from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
