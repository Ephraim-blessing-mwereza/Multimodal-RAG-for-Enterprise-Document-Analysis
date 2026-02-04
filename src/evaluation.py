"""
Evaluation framework for the RAG system.

Metrics:
1. Retrieval Quality (Precision@K, Recall@K, MRR, NDCG)
2. Answer Quality (Faithfulness, Relevance, Completeness)
3. Citation Accuracy
4. Hallucination Rate
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
from loguru import logger

from .models import RetrievedChunk, AnalysisOutput
from .indexing import HybridRAGIndex


class RAGEvaluator:
    """
    Comprehensive evaluation framework for RAG systems.
    
    Evaluates:
    - Retrieval quality
    - Answer faithfulness
    - Citation accuracy
    - Hallucination detection
    """
    
    def __init__(self, rag_index: HybridRAGIndex):
        """
        Initialize the evaluator.
        
        Args:
            rag_index: The RAG index to evaluate
        """
        self.rag_index = rag_index
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        ground_truth: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality with multiple metrics.
        
        Args:
            queries: List of test queries
            ground_truth: List of relevant chunk IDs for each query
            k_values: K values to compute metrics for
            
        Returns:
            Dictionary of metric names to values
        """
        results = {}
        
        for k in k_values:
            precisions = []
            recalls = []
            mrrs = []
            ndcgs = []
            
            for query, relevant_ids in zip(queries, ground_truth):
                if not relevant_ids:
                    continue
                    
                retrieved = self.rag_index.search_hybrid(query, top_k=k)
                retrieved_ids = [r.chunk.chunk_id for r in retrieved]
                
                # Precision@K
                relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
                precision = relevant_retrieved / k if k > 0 else 0
                precisions.append(precision)
                
                # Recall@K
                recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0
                recalls.append(recall)
                
                # MRR (Mean Reciprocal Rank)
                mrr = 0
                for i, rid in enumerate(retrieved_ids):
                    if rid in relevant_ids:
                        mrr = 1 / (i + 1)
                        break
                mrrs.append(mrr)
                
                # NDCG@K
                dcg = self._compute_dcg(retrieved_ids, relevant_ids, k)
                idcg = self._compute_idcg(relevant_ids, k)
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)
            
            if precisions:
                results[f"precision@{k}"] = np.mean(precisions)
                results[f"recall@{k}"] = np.mean(recalls)
            
            if mrrs:
                results["mrr"] = np.mean(mrrs)
            
            if ndcgs:
                results[f"ndcg@{k}"] = np.mean(ndcgs)
        
        return results
    
    def _compute_dcg(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """Compute Discounted Cumulative Gain."""
        dcg = 0.0
        for i, rid in enumerate(retrieved_ids[:k]):
            rel = 1 if rid in relevant_ids else 0
            dcg += rel / np.log2(i + 2)  # +2 because i starts at 0
        return dcg
    
    def _compute_idcg(self, relevant_ids: List[str], k: int) -> float:
        """Compute Ideal DCG."""
        idcg = 0.0
        for i in range(min(len(relevant_ids), k)):
            idcg += 1 / np.log2(i + 2)
        return idcg
    
    def evaluate_answer_faithfulness(
        self,
        answer: str,
        source_chunks: List[RetrievedChunk],
        method: str = "overlap"
    ) -> Dict[str, float]:
        """
        Evaluate if the answer is faithful to source content.
        
        Args:
            answer: Generated answer
            source_chunks: Retrieved context chunks
            method: Evaluation method ('overlap' or 'nli')
            
        Returns:
            Dictionary with faithfulness metrics
        """
        if method == "overlap":
            return self._faithfulness_overlap(answer, source_chunks)
        else:
            # NLI-based would require additional model
            logger.warning("NLI-based faithfulness not implemented, using overlap")
            return self._faithfulness_overlap(answer, source_chunks)
    
    def _faithfulness_overlap(
        self,
        answer: str,
        source_chunks: List[RetrievedChunk]
    ) -> Dict[str, float]:
        """Calculate faithfulness using word overlap."""
        # Tokenize answer
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of',
            'and', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'this', 'that', 'it', 'be', 'have', 'has', 'had',
        }
        answer_words -= stop_words
        
        # Collect source words
        source_words = set()
        for chunk in source_chunks:
            words = set(chunk.chunk.content.lower().split())
            source_words.update(words - stop_words)
        
        if not answer_words:
            return {"faithfulness_score": 1.0, "overlap_ratio": 1.0}
        
        overlap = len(answer_words & source_words)
        overlap_ratio = overlap / len(answer_words)
        
        return {
            "faithfulness_score": overlap_ratio,
            "overlap_ratio": overlap_ratio,
            "answer_terms": len(answer_words),
            "matched_terms": overlap
        }
    
    def evaluate_citation_accuracy(
        self,
        output: AnalysisOutput
    ) -> Dict[str, float]:
        """
        Evaluate citation accuracy in the output.
        
        Args:
            output: Analysis output to evaluate
            
        Returns:
            Dictionary with citation metrics
        """
        import re
        
        # Extract citations from summary
        citation_pattern = r'\[Doc:\s*([^,\]]+)(?:,\s*Page:\s*(\d+))?\]'
        found_citations = re.findall(citation_pattern, output.summary)
        
        # Check against actual citations
        actual_docs = {c['document'] for c in output.citations}
        cited_docs = {c[0].strip() for c in found_citations}
        
        # Calculate metrics
        valid_citations = len(cited_docs & actual_docs)
        total_citations = len(found_citations)
        
        precision = valid_citations / total_citations if total_citations > 0 else 0
        recall = valid_citations / len(actual_docs) if actual_docs else 0
        
        return {
            "citation_precision": precision,
            "citation_recall": recall,
            "total_citations": total_citations,
            "valid_citations": valid_citations,
            "source_documents": len(actual_docs)
        }
    
    def evaluate_hallucination(
        self,
        output: AnalysisOutput
    ) -> Dict[str, float]:
        """
        Evaluate hallucination rate in the output.
        
        Args:
            output: Analysis output to evaluate
            
        Returns:
            Dictionary with hallucination metrics
        """
        total_data = len(output.extracted_data)
        verified_data = sum(
            1 for d in output.extracted_data
            if d.get('verified', False)
        )
        
        unverified_data = total_data - verified_data
        hallucination_rate = unverified_data / total_data if total_data > 0 else 0
        
        return {
            "hallucination_rate": hallucination_rate,
            "total_claims": total_data,
            "verified_claims": verified_data,
            "unverified_claims": unverified_data
        }
    
    def run_evaluation_suite(
        self,
        test_cases: List[Dict[str, Any]],
        agent=None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation suite.
        
        Args:
            test_cases: List of test cases with format:
                {
                    "query": "...",
                    "expected_answer": "...",  # Optional
                    "relevant_chunk_ids": [...]  # Optional
                }
            agent: Optional RAGAgent for answer quality evaluation
            
        Returns:
            Dictionary with all evaluation results
        """
        results = {
            "retrieval_metrics": {},
            "answer_metrics": [],
            "citation_metrics": [],
            "hallucination_metrics": [],
            "aggregate": {}
        }
        
        # Retrieval evaluation
        queries = [tc["query"] for tc in test_cases]
        ground_truth = [tc.get("relevant_chunk_ids", []) for tc in test_cases]
        
        if any(ground_truth):
            results["retrieval_metrics"] = self.evaluate_retrieval(
                queries, ground_truth
            )
        
        # Answer quality evaluation (if agent provided)
        if agent:
            faithfulness_scores = []
            citation_scores = []
            hallucination_rates = []
            
            for tc in test_cases:
                try:
                    output = agent.query(tc["query"])
                    chunks = self.rag_index.search_hybrid(tc["query"])
                    
                    # Faithfulness
                    faith = self.evaluate_answer_faithfulness(
                        output.summary, chunks
                    )
                    faithfulness_scores.append(faith["faithfulness_score"])
                    results["answer_metrics"].append(faith)
                    
                    # Citations
                    cite = self.evaluate_citation_accuracy(output)
                    citation_scores.append(cite["citation_precision"])
                    results["citation_metrics"].append(cite)
                    
                    # Hallucination
                    hall = self.evaluate_hallucination(output)
                    hallucination_rates.append(hall["hallucination_rate"])
                    results["hallucination_metrics"].append(hall)
                    
                except Exception as e:
                    logger.error(f"Evaluation error for query: {tc['query']}: {e}")
            
            # Aggregate scores
            if faithfulness_scores:
                results["aggregate"]["avg_faithfulness"] = np.mean(faithfulness_scores)
            if citation_scores:
                results["aggregate"]["avg_citation_precision"] = np.mean(citation_scores)
            if hallucination_rates:
                results["aggregate"]["avg_hallucination_rate"] = np.mean(hallucination_rates)
        
        return results
    
    def generate_report(
        self,
        evaluation_results: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            evaluation_results: Results from run_evaluation_suite
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "RAG SYSTEM EVALUATION REPORT",
            "=" * 60,
            ""
        ]
        
        # Retrieval metrics
        if evaluation_results.get("retrieval_metrics"):
            lines.append("RETRIEVAL METRICS:")
            lines.append("-" * 40)
            for metric, value in evaluation_results["retrieval_metrics"].items():
                lines.append(f"  {metric}: {value:.4f}")
            lines.append("")
        
        # Aggregate metrics
        if evaluation_results.get("aggregate"):
            lines.append("AGGREGATE METRICS:")
            lines.append("-" * 40)
            for metric, value in evaluation_results["aggregate"].items():
                lines.append(f"  {metric}: {value:.4f}")
            lines.append("")
        
        # Summary
        lines.append("SUMMARY:")
        lines.append("-" * 40)
        
        ret = evaluation_results.get("retrieval_metrics", {})
        agg = evaluation_results.get("aggregate", {})
        
        if ret.get("precision@5"):
            lines.append(f"  Retrieval Quality: {'GOOD' if ret['precision@5'] > 0.6 else 'NEEDS IMPROVEMENT'}")
        if agg.get("avg_faithfulness"):
            lines.append(f"  Answer Faithfulness: {'HIGH' if agg['avg_faithfulness'] > 0.7 else 'MEDIUM' if agg['avg_faithfulness'] > 0.4 else 'LOW'}")
        if agg.get("avg_hallucination_rate"):
            lines.append(f"  Hallucination Risk: {'LOW' if agg['avg_hallucination_rate'] < 0.2 else 'MEDIUM' if agg['avg_hallucination_rate'] < 0.4 else 'HIGH'}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
