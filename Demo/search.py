from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple
import re
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document chunking and preprocessing"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def create_chunks(self, text: str) -> List[Dict[str, any]]:
        """
        Create semantic chunks from document text with metadata
        """
        # Clean and preprocess text
        text = self._clean_text(text)
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        # Create overlapping chunks
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            # If adding this sentence would exceed chunk size, finalize current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'start_sentence': max(0, i - len(current_chunk.split('.'))),
                    'end_sentence': i,
                    'length': len(current_chunk)
                })
                
                # Start new chunk with overlap
                sentences_in_chunk = current_chunk.split('.')
                overlap_sentences = sentences_in_chunk[-self.chunk_overlap//50:] if len(sentences_in_chunk) > self.chunk_overlap//50 else sentences_in_chunk
                current_chunk = '. '.join(overlap_sentences) + '. ' if overlap_sentences else ""
                chunk_id += 1
            
            current_chunk += sentence + ". "
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'start_sentence': max(0, len(sentences) - len(current_chunk.split('.'))),
                'end_sentence': len(sentences),
                'length': len(current_chunk)
            })
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep periods and necessary punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]]', '', text)
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - could be improved with NLTK
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

class SemanticSearch:
    """Handles semantic search using sentence transformers and FAISS"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    
    def build_index(self, chunks: List[Dict[str, any]]) -> Tuple[faiss.Index, np.ndarray]:
        """
        Build FAISS index from document chunks
        """
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings for chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        logger.info(f"Built FAISS index with {index.ntotal} vectors of dimension {dimension}")
        return index, embeddings
    
    def search(self, index_data: Tuple[faiss.Index, np.ndarray], query: str, 
               chunks: List[Dict[str, any]], top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for most relevant chunks given a query
        """
        index, _ = index_data
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding.astype('float32'), top_k)
        
        # Return chunks with scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(chunks):  # Valid index
                chunk = chunks[idx].copy()
                chunk['relevance_score'] = float(score)
                chunk['rank'] = i + 1
                results.append(chunk)
        
        logger.info(f"Retrieved {len(results)} relevant chunks for query")
        return results
    
    def get_query_keywords(self, query: str) -> List[str]:
        """
        Extract key terms from query for better matching
        """
        # Simple keyword extraction - could be improved with NER/POS tagging
        keywords = []
        
        # Medical/insurance specific terms
        medical_terms = re.findall(r'\b(?:surgery|treatment|condition|disease|coverage|policy|premium|claim|benefit)\b', query.lower())
        keywords.extend(medical_terms)
        
        # Extract capitalized terms (likely important nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', query)
        keywords.extend([term.lower() for term in proper_nouns])
        
        # Extract numbers (amounts, periods, etc.)
        numbers = re.findall(r'\b\d+\b', query)
        keywords.extend(numbers)
        
        return list(set(keywords))