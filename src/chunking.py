"""
Smart chunking strategies for different content types.

Design Principles:
1. Semantic chunking: Split at sentence boundaries, preserve context
2. Table chunking: Keep tables intact or split with preserved headers
3. Hierarchical: Maintain parent-child relationships
4. Overlap: Ensure context continuity across chunks
"""

import re
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from loguru import logger

from .models import DocumentChunk, ContentType


class SmartChunker:
    """
    Intelligent chunking engine for multi-modal documents.
    
    Implements multiple chunking strategies optimized for different
    content types while maintaining semantic coherence.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_table_rows: int = 10,
        min_chunk_size: int = 50
    ):
        """
        Initialize the smart chunker.
        
        Args:
            chunk_size: Target chunk size in words
            chunk_overlap: Number of words to overlap between chunks
            max_table_rows: Maximum rows per table chunk
            min_chunk_size: Minimum chunk size (discard smaller)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_rows = max_table_rows
        self.min_chunk_size = min_chunk_size
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'(?<=[.!?])\s+')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        
        # Section header patterns
        self.section_patterns = [
            re.compile(r'^#{1,6}\s+.+', re.MULTILINE),  # Markdown headers
            re.compile(r'^\d+\.\s+[A-Z].+', re.MULTILINE),  # Numbered sections
            re.compile(r'^[A-Z][A-Z\s]{3,}$', re.MULTILINE),  # ALL CAPS headers
            re.compile(r'^(?:Chapter|Section|Part)\s+\d+', re.MULTILINE | re.IGNORECASE),
        ]
        
        logger.info(
            f"SmartChunker initialized (size={chunk_size}, "
            f"overlap={chunk_overlap}, max_table_rows={max_table_rows})"
        )
    
    def chunk_document(self, document: Dict) -> List[DocumentChunk]:
        """
        Chunk a document into semantic units.
        
        Strategy:
        1. Process text blocks with semantic chunking
        2. Process tables with table-aware chunking
        3. Process images with descriptions
        4. Merge and sort by document order
        
        Args:
            document: Ingested document dictionary
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        document_name = document['document_name']
        
        # Track current section for hierarchy
        current_section = None
        chunk_index = 0
        
        # Process text blocks
        for text_block in document['text_blocks']:
            page_num = text_block['page_number']
            text = text_block['text']
            
            # Detect sections
            section = self._detect_section(text)
            if section:
                current_section = section
            
            # Chunk text semantically
            text_chunks = self._chunk_text_semantic(text)
            
            for chunk_text in text_chunks:
                if chunk_text.strip() and len(chunk_text.split()) >= self.min_chunk_size:
                    chunks.append(DocumentChunk(
                        chunk_id="",
                        content=chunk_text.strip(),
                        content_type=ContentType.TEXT,
                        document_name=document_name,
                        page_number=page_num,
                        parent_section=current_section,
                        chunk_index=chunk_index,
                        metadata={'source': 'text_extraction'}
                    ))
                    chunk_index += 1
        
        # Process tables
        for table_info in document.get('tables', []):
            table_chunks = self._chunk_table(table_info, document_name)
            for chunk in table_chunks:
                chunk.chunk_index = chunk_index
                chunk.parent_section = current_section
                chunks.append(chunk)
                chunk_index += 1
        
        # Process images
        for img_info in document.get('images', []):
            img_chunk = self._create_image_chunk(img_info, document_name)
            if img_chunk:
                img_chunk.chunk_index = chunk_index
                chunks.append(img_chunk)
                chunk_index += 1
        
        logger.info(f"Created {len(chunks)} chunks from {document_name}")
        return chunks
    
    def _chunk_text_semantic(self, text: str) -> List[str]:
        """
        Semantic text chunking with overlap.
        
        Strategy:
        1. Split by paragraphs first
        2. If paragraph > chunk_size, split by sentences
        3. Merge small chunks to reach target size
        4. Add overlap from previous chunk
        """
        if not text.strip():
            return []
        
        # Split by paragraphs
        paragraphs = self.paragraph_pattern.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para_len = len(para.split())
            current_len = len(current_chunk.split())
            
            # If paragraph fits, add to current chunk
            if current_len + para_len <= self.chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph itself is too large, split by sentences
                if para_len > self.chunk_size:
                    sentence_chunks = self._split_by_sentences(para)
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add overlap
        chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split long text by sentences, grouping to target size."""
        sentences = self.sentence_endings.split(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk.split()) + len(sentence.split()) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap from previous chunk for context continuity."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_words = chunks[i-1].split()
            overlap_words = min(self.chunk_overlap, len(prev_words))
            
            if overlap_words > 0:
                overlap_text = " ".join(prev_words[-overlap_words:])
                overlapped.append(f"[...] {overlap_text}\n\n{chunks[i]}")
            else:
                overlapped.append(chunks[i])
        
        return overlapped
    
    def _chunk_table(self, table_info: Dict, doc_name: str) -> List[DocumentChunk]:
        """
        Table-aware chunking strategy.
        
        Design Decisions:
        1. Small tables (< max_rows): Keep intact as single chunk
        2. Large tables: Split by row groups, ALWAYS include headers
        3. Generate text summary for semantic search
        4. Store structured data for precise retrieval
        """
        chunks = []
        df = table_info['dataframe']
        page_num = table_info['page_number']
        
        # Generate table summary for semantic search
        summary = self._generate_table_summary(df)
        
        if len(df) <= self.max_table_rows:
            # Small table: keep intact
            table_text = self._dataframe_to_text(df)
            content = f"TABLE:\n{summary}\n\n{table_text}"
            
            chunks.append(DocumentChunk(
                chunk_id="",
                content=content,
                content_type=ContentType.TABLE,
                document_name=doc_name,
                page_number=page_num,
                table_data=df,
                table_summary=summary,
                metadata={
                    'table_index': table_info['table_index'],
                    'rows': len(df),
                    'columns': list(df.columns),
                    'complete_table': True
                }
            ))
        else:
            # Large table: split by row groups with headers
            headers = df.columns.tolist()
            total_rows = len(df)
            
            for i in range(0, len(df), self.max_table_rows):
                chunk_df = df.iloc[i:i + self.max_table_rows]
                table_text = self._dataframe_to_text(chunk_df)
                
                row_start = i + 1
                row_end = min(i + self.max_table_rows, total_rows)
                row_range = f"rows {row_start}-{row_end}"
                
                content = (
                    f"TABLE ({row_range} of {total_rows}):\n"
                    f"{summary}\n\n"
                    f"Headers: {headers}\n\n"
                    f"{table_text}"
                )
                
                chunks.append(DocumentChunk(
                    chunk_id="",
                    content=content,
                    content_type=ContentType.TABLE,
                    document_name=doc_name,
                    page_number=page_num,
                    table_data=chunk_df,
                    table_summary=summary,
                    metadata={
                        'table_index': table_info['table_index'],
                        'row_range': row_range,
                        'row_start': row_start,
                        'row_end': row_end,
                        'total_rows': total_rows,
                        'columns': headers,
                        'complete_table': False
                    }
                ))
        
        return chunks
    
    def _generate_table_summary(self, df: pd.DataFrame) -> str:
        """Generate a natural language summary of a table."""
        cols = df.columns.tolist()
        rows = len(df)
        
        summary_parts = [
            f"Table with {rows} rows and {len(cols)} columns.",
            f"Columns: {', '.join(str(c) for c in cols)}."
        ]
        
        # Add numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            for col in numeric_cols[:3]:  # Limit to first 3
                try:
                    col_data = pd.to_numeric(df[col], errors='coerce')
                    if col_data.notna().any():
                        stats = (
                            f"{col}: min={col_data.min():.2f}, "
                            f"max={col_data.max():.2f}, "
                            f"mean={col_data.mean():.2f}"
                        )
                        summary_parts.append(stats)
                except Exception:
                    pass
        
        return " ".join(summary_parts)
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to readable text format."""
        try:
            return df.to_markdown(index=False)
        except Exception:
            # Fallback to simple string representation
            return df.to_string(index=False)
    
    def _create_image_chunk(
        self,
        img_info: Dict,
        doc_name: str
    ) -> Optional[DocumentChunk]:
        """Create a chunk for an image with description."""
        # In production, use a vision model to generate descriptions
        description = (
            f"Image on page {img_info['page_number']} "
            f"({img_info['width']}x{img_info['height']} pixels)"
        )
        
        return DocumentChunk(
            chunk_id="",
            content=f"[IMAGE] {description}",
            content_type=ContentType.IMAGE,
            document_name=doc_name,
            page_number=img_info['page_number'],
            image_description=description,
            metadata={
                'image_index': img_info['image_index'],
                'width': img_info['width'],
                'height': img_info['height']
            }
        )
    
    def _detect_section(self, text: str) -> Optional[str]:
        """Detect section headers in text."""
        # Only check the beginning of text
        check_text = text[:500]
        
        for pattern in self.section_patterns:
            match = pattern.search(check_text)
            if match:
                return match.group().strip()
        
        return None
