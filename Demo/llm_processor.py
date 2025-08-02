from openai import OpenAI
import json
from typing import Dict, List, Any
import logging
import os
from pydantic import BaseModel

# Optional tiktoken import
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available - token counting will be estimated")

logger = logging.getLogger(__name__)

class QueryStructure(BaseModel):
    """Structured representation of a parsed query"""
    intent: str
    entities: Dict[str, Any]
    keywords: List[str]
    domain: str
    complexity: str

class LLMProcessor:
    """Handles all LLM interactions for query processing and answer generation"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", max_tokens: int = 1000):
        self.model = model
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client (NEW v1.x format)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI client initialized with model: {model}")
        else:
            self.client = None
            logger.warning("OPENAI_API_KEY not set. LLM features will be limited.")
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query into structured format using LLM
        """
        system_prompt = """You are an expert query parser for insurance, legal, HR, and compliance documents. 
        Parse the given query and extract structured information.
        
        Return a JSON object with:
        - intent: The main intent (e.g., "coverage_check", "eligibility", "claim_amount", "waiting_period")
        - entities: Extracted entities like procedures, amounts, time periods, conditions
        - keywords: Key terms for semantic search
        - domain: Document domain (insurance, legal, hr, compliance)
        - complexity: Query complexity (simple, medium, complex)
        
        Be precise and comprehensive in your extraction."""
        
        user_prompt = f"Parse this query: '{query}'"
        
        try:
            if not self.client:
                # Fallback parsing without LLM
                return self._fallback_parse_query(query)
            
            # NEW: Updated API call for OpenAI v1.x
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Successfully parsed query with intent: {result.get('intent', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM query parsing: {str(e)}")
            return self._fallback_parse_query(query)
    
    def generate_answer(self, question: str, context: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive answer with reasoning using LLM
        """
        # Prepare context from retrieved chunks
        context_text = self._prepare_context(retrieved_chunks)
        
        system_prompt = """You are an expert document analyzer specializing in insurance, legal, HR, and compliance documents.
        
        Given a question and relevant document excerpts, provide:
        1. A clear, accurate answer
        2. Specific clause references that support your answer
        3. Step-by-step reasoning
        4. Confidence level (high/medium/low)
        5. Any important conditions or limitations
        
        Be precise, cite specific clauses, and explain your reasoning clearly.
        If information is insufficient, state what additional information would be needed."""
        
        user_prompt = f"""Question: {question}
        
        Relevant Document Excerpts:
        {context_text}
        
        Provide a comprehensive answer with supporting evidence and reasoning."""
        
        try:
            if not self.client:
                # Fallback when no API key
                return self._fallback_generate_answer(question, retrieved_chunks)
            
            # NEW: Updated API call for OpenAI v1.x
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.2
            )
            
            answer_text = response.choices[0].message.content
            
            # Structure the response
            return {
                "answer": answer_text,
                "reasoning": self._extract_reasoning(answer_text),
                "confidence": self._assess_confidence(retrieved_chunks),
                "supporting_chunks": [chunk['id'] for chunk in retrieved_chunks[:3]],
                "token_usage": response.usage.total_tokens if response.usage else self._estimate_tokens(user_prompt + answer_text)
            }
            
        except Exception as e:
            logger.error(f"Error in LLM answer generation: {str(e)}")
            return self._fallback_generate_answer(question, retrieved_chunks)
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context from retrieved chunks for LLM input"""
        context_parts = []
        for i, chunk in enumerate(chunks[:5]):  # Limit to top 5 chunks
            context_parts.append(f"[Excerpt {i+1}] (Relevance: {chunk.get('relevance_score', 0):.3f})")
            context_parts.append(chunk['text'])
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def _extract_reasoning(self, answer_text: str) -> str:
        """Extract reasoning section from LLM response"""
        # Simple extraction - could be improved with structured prompting
        lines = answer_text.split('\n')
        reasoning_lines = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['because', 'since', 'according to', 'based on', 'reasoning']):
                reasoning_lines.append(line)
        
        return ' '.join(reasoning_lines) if reasoning_lines else "Reasoning not explicitly provided."
    
    def _assess_confidence(self, chunks: List[Dict]) -> str:
        """Assess confidence based on retrieved chunks quality"""
        if not chunks:
            return "low"
        
        avg_score = sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks)
        
        if avg_score > 0.8:
            return "high"
        elif avg_score > 0.6:
            return "medium"
        else:
            return "low"
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count when tiktoken is not available"""
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                return len(encoding.encode(text))
            except:
                pass
        
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def _fallback_parse_query(self, query: str) -> Dict[str, Any]:
        """Fallback query parsing without LLM"""
        import re
        
        # Simple rule-based parsing
        entities = {}
        
        # Extract common patterns
        if re.search(r'surgery|operation', query.lower()):
            entities['procedure_type'] = 'surgery'
        
        if re.search(r'waiting period|wait', query.lower()):
            entities['concern'] = 'waiting_period'
        
        if re.search(r'cover|coverage', query.lower()):
            entities['concern'] = 'coverage'
        
        # Extract numbers (amounts, periods)
        numbers = re.findall(r'\d+', query)
        if numbers:
            entities['numbers'] = numbers
        
        return {
            "intent": "coverage_check",
            "entities": entities,
            "keywords": query.lower().split(),
            "domain": "insurance",
            "complexity": "medium"
        }
    
    def _fallback_generate_answer(self, question: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Fallback answer generation without LLM"""
        if not chunks:
            return {
                "answer": "Unable to find relevant information in the document to answer this question.",
                "reasoning": "No relevant chunks retrieved from the document.",
                "confidence": "low",
                "supporting_chunks": [],
                "token_usage": 0
            }
        
        # Use the most relevant chunk as basis for answer
        best_chunk = chunks[0]
        
        return {
            "answer": f"Based on the document excerpt: {best_chunk['text'][:200]}...",
            "reasoning": f"Answer derived from highest relevance chunk (score: {best_chunk.get('relevance_score', 0):.3f})",
            "confidence": self._assess_confidence(chunks),
            "supporting_chunks": [chunk['id'] for chunk in chunks[:3]],
            "token_usage": 0
        }