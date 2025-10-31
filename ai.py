"""
Astrophysics RAG Chatbot with Groq
Friendly explainer that makes academic concepts easy to understand
"""

import os
import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict, Tuple

# Load environment variables
load_dotenv()


class AstrophysicsRAGChatbot:
    """RAG Chatbot for Astrophysics Research Papers"""
    
    def __init__(
        self,
        index_path: str = "faiss_index/astrophysics.index",
        metadata_path: str = "faiss_index/metadata.pkl",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5
    ):
        """
        Initialize the RAG chatbot
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata pickle file
            embedding_model: Sentence transformer model name
            top_k: Number of relevant chunks to retrieve
        """
        print("🚀 Initializing Astrophysics RAG Chatbot...")
        
        # Load Groq API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")
        
        # Initialize Groq client
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile"  # Latest and most powerful
        print(f"✓ Groq model loaded: {self.model_name}")
        
        # Load embedding model
        print(f"✓ Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Load FAISS index
        print(f"✓ Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        print(f"✓ Loading metadata from {metadata_path}")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.top_k = top_k
        
        # System prompt for friendly explainer personality
        self.system_prompt = """You are a friendly astrophysics research assistant who makes complex scientific concepts easy to understand. 

Your personality:
- Explain academic concepts in simple, everyday language
- Use analogies and examples when helpful
- Break down complex ideas into digestible parts
- Be enthusiastic about the science!
- Avoid unnecessary jargon, but when you must use technical terms, explain them
- Be conversational and warm in tone

When answering:
1. Use the provided research paper context to answer accurately
2. Explain complex concepts in simple terms
3. If the context doesn't contain the answer, say so honestly
4. Always cite which papers you're referencing
5. Be concise but thorough

Remember: Your goal is to make astrophysics accessible and exciting for everyone!"""
        
        print(f"✓ Chatbot ready! Loaded {len(self.metadata):,} chunks from research papers")
        print("="*60)
    
    def retrieve_context(self, query: str) -> Tuple[List[str], List[Dict]]:
        """
        Retrieve relevant context from FAISS index
        
        Args:
            query: User's question
            
        Returns:
            Tuple of (context_texts, source_metadata)
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, self.top_k)
        
        # Get context and metadata
        context_texts = []
        sources = []
        
        for idx, score in zip(indices[0], distances[0]):
            # Get metadata for this chunk
            meta = self.metadata[idx]
            
            # Load the actual text from chunked_papers.json
            # (We need to do this since metadata doesn't contain full text)
            # For now, we'll use a placeholder - you can optimize this later
            
            sources.append({
                'paper_id': meta['paper_id'],
                'section': meta['section'],
                'score': float(score),
                'chunk_index': meta['chunk_index']
            })
        
        return context_texts, sources
    
    def retrieve_context_with_text(self, query: str, chunks_file: str = "chunked_papers.json") -> Tuple[List[str], List[Dict]]:
        """
        Retrieve relevant context with actual text from chunks file
        
        Args:
            query: User's question
            chunks_file: Path to chunked papers JSON
            
        Returns:
            Tuple of (context_texts, source_metadata)
        """
        import json
        
        # Load chunks file (we'll cache this in production)
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        chunks = data['chunks']
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, self.top_k)
        
        # Get context and metadata
        context_texts = []
        sources = []
        
        for idx, score in zip(indices[0], distances[0]):
            chunk = chunks[idx]
            context_texts.append(chunk['text'])
            
            sources.append({
                'paper_id': chunk['metadata']['paper_id'],
                'section': chunk['metadata']['section'],
                'score': float(score),
                'chunk_index': chunk['metadata']['chunk_index'],
                'source_file': chunk['metadata']['source_file']
            })
        
        return context_texts, sources
    
    def format_context(self, context_texts: List[str], sources: List[Dict]) -> str:
        """
        Format retrieved context for the LLM
        
        Args:
            context_texts: List of relevant text chunks
            sources: List of source metadata
            
        Returns:
            Formatted context string
        """
        formatted = "RELEVANT RESEARCH CONTEXT:\n\n"
        
        for i, (text, source) in enumerate(zip(context_texts, sources), 1):
            formatted += f"[Source {i} - Paper: {source['paper_id']}, Section: {source['section']}]\n"
            formatted += f"{text}\n\n"
        
        return formatted
    
    def format_sources(self, sources: List[Dict]) -> str:
        """
        Format sources for citation
        
        Args:
            sources: List of source metadata
            
        Returns:
            Formatted sources string
        """
        formatted = "\n\n📚 **Sources:**\n"
        
        # Group by paper_id
        papers = {}
        for source in sources:
            paper_id = source['paper_id']
            if paper_id not in papers:
                papers[paper_id] = []
            papers[paper_id].append(source['section'])
        
        for i, (paper_id, sections) in enumerate(papers.items(), 1):
            unique_sections = list(set(sections))
            sections_str = ", ".join(unique_sections[:3])  # Show max 3 sections
            formatted += f"{i}. **{paper_id}** (Sections: {sections_str})\n"
        
        return formatted
    
    def ask(self, question: str, include_sources: bool = False) -> Dict:
        """
        Ask a question and get an answer with sources
        
        Args:
            question: User's question
            include_sources: Whether to include source citations
            
        Returns:
            Dictionary with 'answer', 'sources', and 'context'
        """
        # Retrieve relevant context
        context_texts, sources = self.retrieve_context_with_text(question)
        
        # Format context
        context = self.format_context(context_texts, sources)
        
        # Build prompt
        prompt = f"""{self.system_prompt}

{context}

USER QUESTION: {question}

Please provide a friendly, easy-to-understand answer based on the research context above. Remember to explain complex concepts in simple terms!"""
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{context}\n\nUSER QUESTION: {question}\n\nPlease provide a friendly, easy-to-understand answer based on the research context above. Remember to explain complex concepts in simple terms!"}
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content
            
            # Add sources if requested
            if include_sources:
                answer += self.format_sources(sources)
            
            return {
                'answer': answer,
                'sources': sources,
                'context': context_texts,
                'success': True
            }
            
        except Exception as e:
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'sources': [],
                'context': [],
                'success': False,
                'error': str(e)
            }
    
    def stream_ask(self, question: str, include_sources: bool = False):
        """
        Ask a question with streaming response (for Streamlit)
        
        Args:
            question: User's question
            include_sources: Whether to include source citations
            
        Yields:
            Chunks of the response text
        """
        # Retrieve relevant context
        context_texts, sources = self.retrieve_context_with_text(question)
        
        # Format context
        context = self.format_context(context_texts, sources)
        
        # Build prompt
        prompt = f"""{self.system_prompt}

{context}

USER QUESTION: {question}

Please provide a friendly, easy-to-understand answer based on the research context above. Remember to explain complex concepts in simple terms!"""
        
        # Generate streaming response
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{context}\n\nUSER QUESTION: {question}\n\nPlease provide a friendly, easy-to-understand answer based on the research context above. Remember to explain complex concepts in simple terms!"}
                ],
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            # Add sources at the end
            if include_sources:
                yield self.format_sources(sources)
                
        except Exception as e:
            yield f"\n\n❌ Error: {str(e)}"


# Simple command-line interface for testing
def main():
    """Test the chatbot in command line"""
    print("="*60)
    print("🌌 Astrophysics RAG Chatbot")
    print("="*60)
    
    # Initialize chatbot
    chatbot = AstrophysicsRAGChatbot()
    
    print("\n💬 Ask me anything about astrophysics!")
    print("(Type 'quit' or 'exit' to stop)\n")
    
    while True:
        # Get user input
        question = input("\n🔭 You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for exploring astrophysics with me! Goodbye!")
            break
        
        if not question:
            continue
        
        # Get answer
        print("\n🤖 Assistant: ", end="", flush=True)
        
        result = chatbot.ask(question)
        
        if result['success']:
            print(result['answer'])
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
