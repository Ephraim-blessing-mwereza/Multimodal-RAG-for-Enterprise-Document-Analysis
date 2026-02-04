"""
Main entry point for the Multimodal RAG system.

Usage:
    python main.py ingest --dir ./data/sample_documents
    python main.py query "What was the revenue in Q3?"
    python main.py evaluate --test-file tests/test_cases.json
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Configure logging
logger.add("logs/rag_{time}.log", rotation="1 day", retention="7 days")


def setup_environment():
    """Load environment variables and configure settings."""
    load_dotenv()
    
    # Check for Gemini key first (preferred), then OpenAI
    api_key = os.getenv("GOOGLE_API_KEY")
    provider = "gemini"
    
    if not api_key or api_key == "your-gemini-api-key-here":
        api_key = os.getenv("OPENAI_API_KEY")
        provider = "openai"
        
    if not api_key or api_key in ["your-api-key-here", "your-gemini-api-key-here"]:
        logger.warning("No API key set (GOOGLE_API_KEY or OPENAI_API_KEY). LLM features will not work.")
        return None, None
    
    return api_key, provider


def run_ingest(args):
    """Run document ingestion pipeline."""
    from src.ingestion import DocumentIngester
    from src.chunking import SmartChunker
    from src.indexing import HybridRAGIndex
    
    logger.info(f"Ingesting documents from: {args.dir}")
    
    # Initialize components
    ingester = DocumentIngester(extract_images=True)
    chunker = SmartChunker(
        chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50))
    )
    index = HybridRAGIndex()
    
    # Ingest documents
    documents = ingester.ingest_directory(Path(args.dir))
    
    if not documents:
        logger.warning("No documents found to ingest.")
        return
    
    # Chunk documents
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
    
    logger.info(f"Total chunks created: {len(all_chunks)}")
    
    # Index chunks
    index.add_chunks(all_chunks)
    
    # Print stats
    stats = index.get_stats()
    logger.info(f"Index stats: {json.dumps(stats, indent=2)}")
    
    return index


def run_query(args, index=None):
    """Run a query against the indexed documents."""
    from src.agent import RAGAgent
    from src.indexing import HybridRAGIndex
    
    api_key, provider = setup_environment()
    if not api_key:
        logger.error("API key required for queries. Set GOOGLE_API_KEY or OPENAI_API_KEY.")
        return
    
    # Use provided index or create new one (would need persistence)
    if index is None:
        logger.warning("No index provided. Running ingest first...")
        # In production, you'd load a persisted index
        ingest_args = argparse.Namespace(dir="./data/sample_documents")
        index = run_ingest(ingest_args)
        
        if index is None:
            return
    
    # Default models based on provider
    default_model = "gemini-1.5-flash" if provider == "gemini" else "gpt-4-turbo-preview"
    
    # Initialize agent
    agent = RAGAgent(
        rag_index=index,
        api_key=api_key,
        model=os.getenv("LLM_MODEL", default_model),
        provider=provider
    )
    
    # Run query
    logger.info(f"Query: {args.query}")
    result = agent.query(args.query)
    
    # Output results
    print("\n" + "="*60)
    print("QUERY RESULT")
    print("="*60)
    print(f"\nSUMMARY:\n{result.summary}")
    
    print(f"\nKEY FINDINGS ({len(result.key_findings)}):")
    for i, finding in enumerate(result.key_findings, 1):
        print(f"  {i}. {finding['finding'][:100]}...")
    
    print(f"\nEXTRACTED DATA ({len(result.extracted_data)}):")
    for data in result.extracted_data:
        verified = "[Y]" if data.get('verified') else "[?]"
        print(f"  {verified} {data['value']}")
    
    print(f"\nCITATIONS ({len(result.citations)}):")
    for cite in result.citations:
        print(f"  - {cite['document']}, Page {cite['page']} (score: {cite['relevance_score']:.3f})")
    
    # Save JSON output
    output_path = Path("outputs") / "latest_result.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(result.to_json())
    logger.info(f"JSON output saved to: {output_path}")
    
    return result


def run_evaluate(args):
    """Run evaluation suite."""
    from src.evaluation import RAGEvaluator
    from src.indexing import HybridRAGIndex
    
    logger.info("Running evaluation suite...")
    
    # Load test cases
    if args.test_file:
        with open(args.test_file) as f:
            test_cases = json.load(f)
    else:
        # Default test cases
        test_cases = [
            {"query": "What was the revenue?"},
            {"query": "What are the risk factors?"},
            {"query": "What is the employee count?"},
        ]
    
    # Initialize index (would need persistence in production)
    ingest_args = argparse.Namespace(dir="./data/sample_documents")
    index = run_ingest(ingest_args)
    
    if index is None:
        logger.error("Failed to build index for evaluation.")
        return
    
    # Run evaluation
    evaluator = RAGEvaluator(index)
    results = evaluator.run_evaluation_suite(test_cases)
    
    # Generate and print report
    report = evaluator.generate_report(results)
    print(report)
    
    # Save results
    output_path = Path("outputs") / "evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Evaluation results saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multimodal RAG for Enterprise Document Analysis"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument(
        "--dir", "-d",
        default="./data/sample_documents",
        help="Directory containing documents to ingest"
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument(
        "query",
        help="Question to ask about the documents"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument(
        "--test-file", "-t",
        help="JSON file with test cases"
    )
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
