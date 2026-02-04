"""
Document ingestion pipeline for multi-modal content extraction.
"""

import io
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pandas as pd
import numpy as np
from loguru import logger


class DocumentIngester:
    """
    Multi-modal document ingestion pipeline.
    
    Extracts text, tables, and images from PDF documents using
    multiple extraction libraries for optimal results.
    
    Features:
        - Text extraction with layout preservation (PyMuPDF)
        - Table extraction with structure detection (pdfplumber)
        - Image extraction with optional OCR
        - Metadata extraction
    """
    
    def __init__(
        self,
        extract_images: bool = True,
        ocr_enabled: bool = False,
        min_image_size: int = 50
    ):
        """
        Initialize the document ingester.
        
        Args:
            extract_images: Whether to extract images from documents
            ocr_enabled: Whether to run OCR on images
            min_image_size: Minimum image dimension to extract (pixels)
        """
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled
        self.min_image_size = min_image_size
        self.supported_formats = ['.pdf']
        
        logger.info(f"DocumentIngester initialized (images={extract_images}, ocr={ocr_enabled})")
    
    def ingest_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Ingest a single document and extract all content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing:
                - document_name: Name of the file
                - file_path: Full path to file
                - text_blocks: List of text content per page
                - tables: List of extracted tables
                - images: List of extracted images
                - metadata: Document metadata
        
        Raises:
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        logger.info(f"Ingesting document: {file_path.name}")
        
        result = {
            'document_name': file_path.name,
            'file_path': str(file_path),
            'text_blocks': [],
            'tables': [],
            'images': [],
            'metadata': {}
        }
        
        # Extract with PyMuPDF for text and images
        result = self._extract_with_pymupdf(file_path, result)
        
        # Extract tables with pdfplumber (better table detection)
        result = self._extract_tables_with_pdfplumber(file_path, result)
        
        logger.info(
            f"Extracted from {file_path.name}: "
            f"{len(result['text_blocks'])} pages, "
            f"{len(result['tables'])} tables, "
            f"{len(result['images'])} images"
        )
        
        return result
    
    def _extract_with_pymupdf(self, file_path: Path, result: Dict) -> Dict:
        """Extract text and images using PyMuPDF."""
        doc = fitz.open(file_path)
        
        # Extract metadata
        result['metadata'] = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'creation_date': doc.metadata.get('creationDate', ''),
            'modification_date': doc.metadata.get('modDate', ''),
            'page_count': len(doc),
            'format': doc.metadata.get('format', 'PDF')
        }
        
        for page_num, page in enumerate(doc):
            # Extract text with layout preservation
            text = page.get_text("text")
            
            # Also get blocks for better structure understanding
            blocks = page.get_text("dict")["blocks"]
            
            result['text_blocks'].append({
                'page_number': page_num + 1,
                'text': text,
                'blocks': blocks,
                'width': page.rect.width,
                'height': page.rect.height
            })
            
            # Extract images if enabled
            if self.extract_images:
                images = page.get_images(full=True)
                for img_idx, img in enumerate(images):
                    extracted = self._extract_image(doc, img, page_num, img_idx)
                    if extracted:
                        result['images'].append(extracted)
        
        doc.close()
        return result
    
    def _extract_image(
        self,
        doc: fitz.Document,
        img: tuple,
        page_num: int,
        img_idx: int
    ) -> Optional[Dict]:
        """Extract a single image from the document."""
        try:
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            
            # Skip small images (likely icons/bullets)
            if pix.width < self.min_image_size or pix.height < self.min_image_size:
                return None
            
            # Convert CMYK to RGB if needed
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            
            return {
                'page_number': page_num + 1,
                'image_index': img_idx,
                'image': pil_image,
                'width': pix.width,
                'height': pix.height,
                'color_space': 'RGB' if pix.n <= 3 else 'RGBA'
            }
        except Exception as e:
            logger.warning(f"Failed to extract image on page {page_num + 1}: {e}")
            return None
    
    def _extract_tables_with_pdfplumber(self, file_path: Path, result: Dict) -> Dict:
        """Extract tables using pdfplumber for better accuracy."""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:  # Has header + data
                            processed = self._process_table(
                                table, page_num, table_idx
                            )
                            if processed:
                                result['tables'].append(processed)
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        return result
    
    def _process_table(
        self,
        table: List[List],
        page_num: int,
        table_idx: int
    ) -> Optional[Dict]:
        """Process a raw table into structured format."""
        try:
            # Use first row as headers
            headers = table[0]
            data = table[1:]
            
            # Clean headers
            clean_headers = []
            for i, h in enumerate(headers):
                if h and str(h).strip():
                    clean_headers.append(str(h).strip())
                else:
                    clean_headers.append(f'col_{i}')
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=clean_headers)
            
            # Clean up empty rows/columns
            df = df.replace('', np.nan)
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            
            if df.empty:
                return None
            
            return {
                'page_number': page_num + 1,
                'table_index': table_idx,
                'dataframe': df,
                'raw_data': table,
                'rows': len(df),
                'columns': list(df.columns),
                'has_numeric': any(
                    pd.api.types.is_numeric_dtype(df[col]) 
                    for col in df.columns
                )
            }
        except Exception as e:
            logger.warning(f"Failed to process table: {e}")
            return None
    
    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = False
    ) -> List[Dict]:
        """
        Ingest all documents in a directory.
        
        Args:
            directory: Path to directory containing documents
            recursive: Whether to search subdirectories
            
        Returns:
            List of ingested document dictionaries
        """
        directory = Path(directory)
        documents = []
        
        pattern = "**/*.pdf" if recursive else "*.pdf"
        
        for file_path in directory.glob(pattern):
            try:
                doc = self.ingest_document(file_path)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path.name}: {e}")
        
        logger.info(f"Ingested {len(documents)} documents from {directory}")
        return documents
