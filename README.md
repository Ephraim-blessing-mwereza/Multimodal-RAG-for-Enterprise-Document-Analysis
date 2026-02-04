# Multimodal RAG for Enterprise Document Analysis

A proof-of-concept RAG (Retrieval Augmented Generation) pipeline for analyzing unstructured and structured enterprise documents including PDFs, tables, and images.

## Overview

This system demonstrates a production-ready approach to document analysis that:

- **Ingests** multiple document types (PDFs with text, tables, images)
- **Chunks** content intelligently based on semantic boundaries and content type
- **Indexes** using hybrid retrieval (dense vectors + sparse BM25)
- **Retrieves** with explainability and relevance filtering
- **Generates** answers with citations and hallucination prevention
- **Outputs** structured JSON with key findings, numerical data, and risk flags

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT INPUT LAYER                            │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                   │
│   │  PDFs   │  │ Tables  │  │ Images  │  │  Logs   │                   │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                   │
└────────┼────────────┼───────────┼────────────┼─────────────────────────┘
         │            │           │            │
         ▼            ▼           ▼            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                               │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│   │ Text Extract │  │Table Extract │  │Image Extract │                 │
│   │  (PyMuPDF)   │  │ (pdfplumber) │  │ (+ OCR)      │                 │
│   └──────────────┘  └──────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        SMART CHUNKING ENGINE                            │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│   │ Semantic Chunker │  │ Table-Aware      │  │ Hierarchical     │     │
│   │ (Sentence-aware) │  │ (Header preserved)│  │ (Section-based) │     │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        HYBRID INDEX LAYER                               │
│   ┌────────────────────────┐      ┌────────────────────────┐           │
│   │     DENSE INDEX        │      │     SPARSE INDEX       │           │
│   │  (sentence-transformers│      │      (BM25)            │           │
│   │   + ChromaDB)          │      │                        │           │
│   └────────────────────────┘      └────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     INTELLIGENT RETRIEVAL                               │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│   │Dense Search  │→ │Hybrid Fusion │→ │ Re-ranking   │                 │
│   │Sparse Search │  │   (RRF)      │  │ + Filtering  │                 │
│   └──────────────┘  └──────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG AGENT                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│   │Context Build │→ │  LLM Gen     │→ │ Validation   │                 │
│   │+ Citations   │  │  (Gemini)    │  │ + Fact Check │                 │
│   └──────────────┘  └──────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                    │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│   │  Summary    │  │ JSON Output │  │  Citations  │  │ Risk Flags  │   │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

See `docs/architecture.md` for detailed Mermaid diagrams.

## Quick Start

### Option A: Automated Setup (Recommended for Windows)

```powershell
# Run the setup script (PowerShell)
.\setup.ps1

# Or use the batch file
.\setup.bat
```

This will:
- Create a virtual environment
- Install all dependencies
- Set up a Jupyter kernel named "Multimodal RAG (Python)"
- Create a `.env` file from the template

### Option B: Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=multimodal-rag --display-name="Multimodal RAG (Python)"
```

### Configuration

```bash
# Copy environment template
copy .env.example .env   # Windows
# cp .env.example .env   # Mac/Linux

# Edit .env and add your Google Gemini API key
GOOGLE_API_KEY=your-gemini-api-key-here
```

**Get a Gemini API Key**: https://makersuite.google.com/app/apikey (Free tier available!)

### Run the Demo

```bash
# Make sure venv is activated, then:

# Option 1: Jupyter Notebook (recommended for demo)
jupyter notebook notebooks/multimodal_rag_demo.ipynb
# Select kernel: "Multimodal RAG (Python)"

# Option 2: Command Line
python main.py ingest --dir ./data/sample_documents
python main.py query "What are the key findings?"

# Option 3: Google Colab
# Upload the notebook and run the install cell
```

### 4. Add Your Documents

Place your PDF files in:
```
data/sample_documents/
```

The system supports:
- Text-heavy reports
- Table-rich documents (financial reports, specifications)
- Mixed content with images

## Project Structure

```
Ray AI/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── docs/
│   └── architecture.md          # Mermaid architecture diagrams
├── notebooks/
│   └── multimodal_rag_demo.ipynb    # Main demo notebook
├── src/
│   ├── __init__.py
│   ├── ingestion.py             # Document ingestion pipeline
│   ├── chunking.py              # Smart chunking strategies
│   ├── indexing.py              # Hybrid RAG index
│   ├── retrieval.py             # Retrieval with explainability
│   ├── agent.py                 # RAG agent with synthesis
│   └── evaluation.py            # Evaluation framework
├── data/
│   └── sample_documents/        # Place your PDFs here
└── outputs/                     # Generated outputs
```

## Key Design Decisions

### 1. Chunking Strategy for Tables

**Problem**: Tables contain structured data that loses meaning when arbitrarily split.

**Solution**:
- **Small tables** (< 10 rows): Keep intact as single chunk
- **Large tables**: Split by row groups, **always preserve headers** in each chunk
- **Generate text summaries** for semantic search (columns, stats, data types)
- **Store structured data** alongside text for precise extraction

```python
# Example: Table chunk structure
{
    "content": "TABLE (rows 1-10 of 50): Revenue by Region\nHeaders: [Region, Q1, Q2, Q3, Q4]\n...",
    "table_data": <DataFrame>,
    "table_summary": "Table with 50 rows and 5 columns. Columns: Region, Q1, Q2, Q3, Q4..."
}
```

### 2. Retrieval Decision Process

**Hybrid Approach**: Combine semantic (dense) and keyword (sparse) retrieval.

```
Query → [Dense Search] → Top-K candidates (semantic similarity)
     → [Sparse Search] → Top-K candidates (keyword matching)
     → [RRF Fusion]    → Merged ranking
     → [Re-ranking]    → Cross-encoder scoring
     → [Filtering]     → Relevance threshold
     → Final context with explanations
```

**Why hybrid?**
- Dense retrieval excels at semantic understanding ("financial performance" → "revenue growth")
- Sparse retrieval excels at exact matches (specific numbers, proper nouns)
- RRF fusion captures benefits of both without tuning weights

### 3. Hallucination Prevention

**Multi-layer approach**:

1. **Prompt Engineering**: System prompt explicitly forbids generating information not in context
2. **Low Temperature**: Use temperature=0.1 for factual accuracy
3. **Citation Requirements**: Force citations for every claim
4. **Post-Generation Validation**:
   - Verify numerical data appears in source chunks
   - Flag claims without supporting evidence
   - Cross-reference extracted entities

```python
# Validation check
def validate_response(output, source_chunks):
    all_content = " ".join(chunk.content for chunk in source_chunks)
    for data_point in output.extracted_data:
        if data_point.value not in all_content:
            data_point.verified = False
            data_point.warning = "Could not verify in source documents"
```

### 4. Scaling for Millions of Documents

**Current (PoC)**: In-memory ChromaDB, single-node processing

**Production Architecture**:

| Component | PoC Solution | Production Solution |
|-----------|-------------|---------------------|
| Vector DB | ChromaDB (in-memory) | Pinecone / Milvus / Weaviate (distributed) |
| Ingestion | Sequential | Celery workers + Kafka queue |
| Embeddings | CPU | GPU inference servers |
| LLM | Gemini API | Azure OpenAI / vLLM / TGI |
| Caching | None | Redis (query + embedding cache) |
| Storage | Local filesystem | S3 / MinIO |

**Scaling Strategies**:
- **Sharding**: Partition by document date, category, or tenant
- **Tiered Retrieval**: Coarse filter (metadata) → Fine search (vectors)
- **Approximate Search**: Use HNSW with larger ef values for accuracy
- **Batch Processing**: Index documents in batches during off-peak hours

### 5. Enterprise / On-Prem Adjustments

| Concern | Cloud Approach | On-Prem Approach |
|---------|---------------|------------------|
| LLM | OpenAI API | Local LLaMA/Mistral via vLLM or Ollama |
| Embeddings | OpenAI embeddings | sentence-transformers (already local) |
| Vector DB | Managed Pinecone | Self-hosted Milvus/Weaviate |
| Document Storage | S3 | MinIO or NFS |
| Auth | OAuth/SAML | LDAP/Active Directory integration |
| Data Isolation | Multi-tenant namespaces | Physical isolation per client |
| Audit | Cloud logging | ELK stack with tamper-proof logs |
| Encryption | AWS KMS | HashiCorp Vault |

## Output Schema

```json
{
  "query": "What was the revenue in Q3?",
  "summary": "The total revenue in Q3 2024 was $45.2 million...",
  "key_findings": [
    {
      "finding": "Revenue increased 12% year-over-year",
      "confidence": "high",
      "sources": ["q3_report.pdf, page 1"]
    }
  ],
  "extracted_data": [
    {
      "metric_name": "Total Revenue",
      "value": 45.2,
      "unit": "million USD",
      "source_location": "page 1, paragraph 2",
      "verified": true
    }
  ],
  "risk_flags": [
    {
      "type": "supply_chain",
      "description": "Supply chain disruptions impacting delivery",
      "severity": "medium",
      "evidence": ["q3_report.pdf, page 1"]
    }
  ],
  "citations": [
    {
      "document": "q3_report.pdf",
      "page": 1,
      "content_preview": "Total Revenue: $45.2 million...",
      "relevance_score": 0.89,
      "retrieval_method": "hybrid"
    }
  ],
  "metadata": {
    "model": "gemini-2.5-flash",
    "chunks_used": 5,
    "validation_performed": true
  }
}
```

## Evaluation Metrics

The system includes evaluation for:

1. **Retrieval Quality**
   - Precision@K: Relevant documents in top-K
   - Recall@K: Coverage of relevant documents
   - MRR: Mean Reciprocal Rank

2. **Answer Quality**
   - Faithfulness: Are claims supported by sources?
   - Relevance: Does the answer address the query?
   - Completeness: Are all aspects covered?

3. **System Quality**
   - Citation accuracy
   - Hallucination rate
   - Response latency



## Future Enhancements

- [ ] Vision model integration (GPT-4V / Claude 3) for image understanding
- [ ] Multi-document reasoning with knowledge graphs
- [ ] Fine-tuned embedding models for domain-specific retrieval
- [ ] Streaming responses for better UX
- [ ] Active learning for retrieval improvement
- [ ] Document classification and auto-routing

## License

MIT License - See LICENSE file for details.
