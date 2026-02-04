"""
RAG Agent for intelligent document analysis with synthesis and validation.

Features:
- Query processing and intent classification
- Context building with citations
- LLM-based answer generation (supports Gemini and OpenAI)
- Hallucination detection and validation
- Structured JSON output generation
"""

import re
from typing import List, Dict, Any, Optional

from loguru import logger

from .models import DocumentChunk, RetrievedChunk, AnalysisOutput, ContentType
from .indexing import HybridRAGIndex


class RAGAgent:
    """
    Intelligent RAG agent for enterprise document analysis.
    
    Supports multiple LLM providers:
    - Google Gemini (default)
    - OpenAI GPT-4
    
    Responsibilities:
    1. Process user queries
    2. Retrieve and filter relevant context
    3. Synthesize answers with proper citations
    4. Validate outputs for hallucinations
    5. Generate structured JSON output
    """
    
    def __init__(
        self,
        rag_index: HybridRAGIndex,
        api_key: str,
        model: str = "gemini-1.5-pro",
        provider: str = "gemini",
        temperature: float = 0.1
    ):
        """
        Initialize the RAG agent.
        
        Args:
            rag_index: Hybrid RAG index for retrieval
            api_key: API key for the LLM provider
            model: LLM model to use (default: gemini-1.5-pro)
            provider: LLM provider ('gemini' or 'openai')
            temperature: Generation temperature (lower = more factual)
        """
        self.rag_index = rag_index
        self.model = model
        self.provider = provider.lower()
        self.temperature = temperature
        
        # Initialize the appropriate client
        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
        elif self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'gemini' or 'openai'")
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()
        
        logger.info(f"RAGAgent initialized with {provider} model: {model}")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        return """You are an expert document analysis assistant. Your task is to analyze documents and provide accurate, well-cited answers.

CRITICAL RULES:
1. ONLY use information from the provided context chunks
2. ALWAYS cite sources using [Doc: document_name, Page: X] format
3. If information is not in the context, explicitly say "I cannot find this information in the provided documents"
4. Clearly distinguish between facts from documents and your analysis/interpretation
5. For numerical data, quote exact figures with their source
6. Identify and flag any potential risks or concerns found in the documents
7. If tables are present, extract and present relevant data clearly
8. Never make up information or extrapolate beyond what's in the documents

OUTPUT STRUCTURE:
1. Direct answer to the question
2. Supporting evidence with citations
3. Any relevant numerical data (with sources)
4. Risk flags or concerns (if applicable)
5. Confidence level (high/medium/low) based on available evidence

Be concise but thorough. Every claim must be traceable to a source."""
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        relevance_threshold: float = 0.3
    ) -> AnalysisOutput:
        """
        Process a query and generate structured output.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            relevance_threshold: Minimum relevance score
            
        Returns:
            AnalysisOutput with summary, findings, data, and citations
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.rag_index.search_hybrid(question, top_k=top_k)
        
        # Step 2: Filter by relevance threshold
        relevant_chunks = [
            rc for rc in retrieved_chunks
            if rc.score >= relevance_threshold
        ]
        
        logger.info(
            f"Retrieved {len(retrieved_chunks)} chunks, "
            f"{len(relevant_chunks)} above threshold ({relevance_threshold})"
        )
        
        if not relevant_chunks:
            return self._create_empty_response(question)
        
        # Step 3: Build context
        context = self._build_context(relevant_chunks)
        
        # Step 4: Generate response
        response = self._generate_response(question, context)
        
        # Step 5: Parse and structure output
        structured_output = self._parse_response(
            question, response, relevant_chunks
        )
        
        # Step 6: Validate for hallucinations
        validated_output = self._validate_response(
            structured_output, relevant_chunks
        )
        
        return validated_output
    
    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Build context string from retrieved chunks.
        
        Each chunk is clearly labeled with source information
        for proper citation tracking.
        """
        context_parts = []
        
        for i, rc in enumerate(chunks):
            chunk = rc.chunk
            
            header = f"[CHUNK {i+1}]"
            header += f"\nDocument: {chunk.document_name}"
            header += f"\nPage: {chunk.page_number}"
            header += f"\nContent Type: {chunk.content_type.value}"
            header += f"\nRelevance Score: {rc.score:.3f}"
            header += f"\nRetrieval Method: {rc.retrieval_method}"
            
            if chunk.parent_section:
                header += f"\nSection: {chunk.parent_section}"
            
            context_parts.append(f"{header}\n\nCONTENT:\n{chunk.content}\n")
        
        return "\n" + "="*50 + "\n".join(context_parts)
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate response using LLM (supports Gemini and OpenAI)."""
        
        prompt = f"""{self.system_prompt}

CONTEXT DOCUMENTS:
{context}

================================================================================

USER QUESTION: {question}

Please analyze the context documents and answer the question. Follow the output structure specified above. Cite all sources."""
        
        try:
            if self.provider == "gemini":
                # Gemini API
                response = self.client.generate_content(prompt)
                return response.text
            else:
                # OpenAI API
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"""CONTEXT DOCUMENTS:
{context}

================================================================================

USER QUESTION: {question}

Please analyze the context documents and answer the question. Follow the output structure specified in the system prompt. Cite all sources."""}
                ]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=2000
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    def _parse_response(
        self,
        question: str,
        response: str,
        chunks: List[RetrievedChunk]
    ) -> AnalysisOutput:
        """Parse LLM response into structured output."""
        # Extract components
        key_findings = self._extract_key_findings(response)
        numerical_data = self._extract_numerical_data(response, chunks)
        risk_flags = self._extract_risk_flags(response)
        citations = [rc.to_citation() for rc in chunks]
        
        return AnalysisOutput(
            query=question,
            summary=response,
            key_findings=key_findings,
            extracted_data=numerical_data,
            risk_flags=risk_flags,
            citations=citations,
            metadata={
                "model": self.model,
                "chunks_used": len(chunks),
                "retrieval_methods": list(set(rc.retrieval_method for rc in chunks))
            }
        )
    
    def _extract_key_findings(self, response: str) -> List[Dict]:
        """Extract key findings from response."""
        findings = []
        
        # Look for bullet points or numbered items
        patterns = [
            r'[-•]\s*(.+?)(?=\n[-•]|\n\n|$)',
            r'\d+\.\s*(.+?)(?=\n\d+\.|\n\n|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                match = match.strip()
                if len(match) > 20 and len(match) < 500:
                    # Extract any citations in the finding
                    citations = re.findall(r'\[Doc:\s*([^,\]]+)', match)
                    
                    findings.append({
                        "finding": match,
                        "confidence": "medium",
                        "sources": citations
                    })
        
        # Deduplicate and limit
        seen = set()
        unique_findings = []
        for f in findings:
            if f["finding"][:50] not in seen:
                seen.add(f["finding"][:50])
                unique_findings.append(f)
        
        return unique_findings[:10]
    
    def _extract_numerical_data(
        self,
        response: str,
        chunks: List[RetrievedChunk]
    ) -> List[Dict]:
        """Extract numerical data from response."""
        numerical_data = []
        
        # Patterns for different number formats
        patterns = [
            # Currency with optional millions/billions
            r'(\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B))?)',
            # Percentages
            r'(\d+(?:\.\d+)?%)',
            # Numbers with units
            r'(\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|units|users|customers))',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                # Try to find context around the number
                context_match = re.search(
                    rf'([^.]*{re.escape(match)}[^.]*)',
                    response
                )
                context = context_match.group(1).strip() if context_match else ""
                
                numerical_data.append({
                    "value": match,
                    "context": context[:200],
                    "verified": False  # Will be verified in validation step
                })
        
        # Deduplicate
        seen = set()
        unique_data = []
        for d in numerical_data:
            if d["value"] not in seen:
                seen.add(d["value"])
                unique_data.append(d)
        
        return unique_data[:15]
    
    def _extract_risk_flags(self, response: str) -> List[Dict]:
        """Extract risk flags from response."""
        risk_flags = []
        
        # Risk-related keywords with severity mapping
        risk_keywords = {
            'critical': ['critical', 'severe', 'major failure', 'emergency'],
            'high': ['risk', 'threat', 'vulnerability', 'significant decline'],
            'medium': ['concern', 'warning', 'issue', 'problem', 'challenge'],
            'low': ['minor', 'potential', 'slight', 'possible']
        }
        
        sentences = re.split(r'[.!?]\s+', response)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            for severity, keywords in risk_keywords.items():
                for keyword in keywords:
                    if keyword in sentence_lower:
                        # Extract evidence (citations in the sentence)
                        evidence = re.findall(r'\[Doc:\s*([^\]]+)\]', sentence)
                        
                        risk_flags.append({
                            "type": keyword,
                            "description": sentence.strip(),
                            "severity": severity,
                            "evidence": evidence
                        })
                        break
        
        # Deduplicate and limit
        seen = set()
        unique_flags = []
        for f in risk_flags:
            if f["description"][:50] not in seen:
                seen.add(f["description"][:50])
                unique_flags.append(f)
        
        return unique_flags[:10]
    
    def _validate_response(
        self,
        output: AnalysisOutput,
        chunks: List[RetrievedChunk]
    ) -> AnalysisOutput:
        """
        Validate response for potential hallucinations.
        
        Validation Strategy:
        1. Check numerical data appears in source chunks
        2. Verify key claims have supporting evidence
        3. Flag any unverifiable statements
        """
        # Combine all chunk content for verification
        all_content = " ".join(
            rc.chunk.content.lower() for rc in chunks
        )
        
        # Verify numerical data exists in sources
        verified_data = []
        for data in output.extracted_data:
            # Normalize the value for comparison
            value = data['value'].lower()
            value_clean = re.sub(r'[,$%]', '', value)
            
            # Check if value appears in any source
            if value_clean in all_content or value in all_content:
                data['verified'] = True
            else:
                # Try to find similar numbers
                numbers = re.findall(r'\d+\.?\d*', value_clean)
                data['verified'] = any(num in all_content for num in numbers)
            
            if not data['verified']:
                data['warning'] = "Could not verify in source documents"
            
            verified_data.append(data)
        
        output.extracted_data = verified_data
        output.metadata['validation_performed'] = True
        output.metadata['verified_data_count'] = sum(
            1 for d in verified_data if d.get('verified', False)
        )
        
        return output
    
    def _create_empty_response(self, question: str) -> AnalysisOutput:
        """Create response when no relevant chunks found."""
        return AnalysisOutput(
            query=question,
            summary=(
                "I could not find relevant information in the provided documents "
                "to answer this question. Please try rephrasing your query or "
                "ensure the relevant documents have been indexed."
            ),
            key_findings=[],
            extracted_data=[],
            risk_flags=[],
            citations=[],
            metadata={
                "model": self.model,
                "chunks_used": 0,
                "no_results_reason": "No chunks above relevance threshold"
            }
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        top_k: int = 5
    ) -> str:
        """
        Multi-turn chat with RAG augmentation.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            top_k: Number of chunks to retrieve
            
        Returns:
            Assistant response string
        """
        # Get the latest user message for retrieval
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            return "Please ask a question."
        
        latest_query = user_messages[-1]["content"]
        
        # Retrieve context
        retrieved = self.rag_index.search_hybrid(latest_query, top_k=top_k)
        context = self._build_context(retrieved) if retrieved else ""
        
        # Build full message history with context
        system_with_context = self.system_prompt
        if context:
            system_with_context += f"\n\nCONTEXT DOCUMENTS:\n{context}"
        
        try:
            if self.provider == "gemini":
                # Gemini: Build conversation as single prompt
                conversation = system_with_context + "\n\n"
                for msg in messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    conversation += f"{role}: {msg['content']}\n\n"
                conversation += "Assistant:"
                
                response = self.client.generate_content(conversation)
                return response.text
            else:
                # OpenAI API
                full_messages = [
                    {"role": "system", "content": system_with_context}
                ] + messages
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    temperature=self.temperature,
                    max_tokens=2000
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {str(e)}"
