"""
Chunking script for Astrophysics Research Papers RAG System

This script processes astrophysics research papers (text files) and creates
semantically meaningful chunks for RAG-based retrieval with the following features:
- Recursive text splitting (sections -> paragraphs -> sentences)
- Token-based chunking with configurable size and overlap
- Metadata preservation (paper ID, section, chunk index)
- LaTeX artifact cleaning
- JSON output format compatible with FAISS embeddings
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import tiktoken


@dataclass
class ChunkMetadata:
    """Metadata for each chunk"""
    paper_id: str
    chunk_index: int
    total_chunks: int
    section: str
    char_count: int
    token_count: int
    source_file: str


@dataclass
class DocumentChunk:
    """Represents a single chunk of text with metadata"""
    text: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'text': self.text,
            'metadata': asdict(self.metadata)
        }


class AstrophysicsChunker:
    """Handles chunking of astrophysics research papers"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the chunker
        
        Args:
            chunk_size: Target chunk size in tokens (default: 512)
            chunk_overlap: Overlap between chunks in tokens (default: 100, ~20%)
            encoding_name: Tiktoken encoding to use for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Section headers pattern (common in research papers)
        self.section_patterns = [
            r'^abstract\s*$',
            r'^introduction\s*$',
            r'^methods?\s*$',
            r'^methodology\s*$',
            r'^results?\s*$',
            r'^discussion\s*$',
            r'^conclusion\s*$',
            r'^references?\s*$',
            r'^acknowledgments?\s*$',
            r'^\d+\.?\s+[A-Z]',  # Numbered sections like "1. Introduction"
            r'^[A-Z][A-Z\s]{3,}$',  # ALL CAPS HEADERS
        ]
    
    def clean_latex_artifacts(self, text: str) -> str:
        """
        Remove LaTeX commands and formatting artifacts
        
        Args:
            text: Raw text with LaTeX artifacts
            
        Returns:
            Cleaned text
        """
        # Remove LaTeX commands at the start
        text = re.sub(r'^\[.*?\].*?$', '', text, flags=re.MULTILINE)
        
        # Remove common LaTeX commands
        latex_commands = [
            r'\\usepackage\{.*?\}',
            r'\\documentclass\{.*?\}',
            r'\\begin\{.*?\}',
            r'\\end\{.*?\}',
            r'\\section\*?\{',
            r'\\subsection\*?\{',
            r'\\cite\{',
            r'\\ref\{',
            r'\\label\{',
            r'\\',
        ]
        
        for pattern in latex_commands:
            text = re.sub(pattern, '', text)
        
        # Remove excessive brackets
        text = re.sub(r'\{|\}', '', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove multiple newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove lines that are just commands or garbage
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that are just LaTeX artifacts or too short
            if len(line) < 3 or line.startswith('[') or line == 'document':
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def identify_section(self, text: str) -> str:
        """
        Identify the section header from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Section name or "unknown"
        """
        lines = text.strip().split('\n')[:5]  # Check first 5 lines
        
        for line in lines:
            line_lower = line.strip().lower()
            for pattern in self.section_patterns:
                if re.match(pattern, line_lower, re.IGNORECASE):
                    return line.strip()[:50]  # Return first 50 chars of section
        
        return "unknown"
    
    def split_by_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into sections based on headers
        
        Returns:
            List of (section_name, section_text) tuples
        """
        sections = []
        current_section = "introduction"
        current_text = []
        
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.strip().lower()
            is_header = False
            
            # Check if line is a section header
            for pattern in self.section_patterns:
                if re.match(pattern, line_lower, re.IGNORECASE) and len(line.strip()) < 100:
                    is_header = True
                    # Save previous section
                    if current_text:
                        sections.append((current_section, '\n'.join(current_text)))
                    current_section = line.strip()[:50]
                    current_text = []
                    break
            
            if not is_header:
                current_text.append(line)
        
        # Add last section
        if current_text:
            sections.append((current_section, '\n'.join(current_text)))
        
        return sections if sections else [("unknown", text)]
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with nltk/spacy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def recursive_split(self, text: str, section_name: str = "unknown") -> List[str]:
        """
        Recursively split text into chunks
        
        Strategy:
        1. Try to keep sections together if they fit
        2. If section is too large, split by paragraphs
        3. If paragraph is too large, split by sentences
        4. If sentence is too large, split by tokens
        
        Args:
            text: Text to split
            section_name: Name of the section
            
        Returns:
            List of text chunks
        """
        token_count = self.count_tokens(text)
        
        # If text fits in chunk size, return as is
        if token_count <= self.chunk_size:
            return [text]
        
        chunks = []
        
        # Try splitting by paragraphs first
        paragraphs = self.split_by_paragraphs(text)
        
        if len(paragraphs) == 1:
            # Single paragraph, try sentences
            sentences = self.split_by_sentences(text)
            chunks = self._combine_small_chunks(sentences)
        else:
            # Multiple paragraphs, combine them into chunks
            chunks = self._combine_small_chunks(paragraphs)
        
        return chunks
    
    def _combine_small_chunks(self, segments: List[str]) -> List[str]:
        """
        Combine small segments into chunks of appropriate size with overlap
        
        Args:
            segments: List of text segments (paragraphs or sentences)
            
        Returns:
            List of combined chunks
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for segment in segments:
            segment_tokens = self.count_tokens(segment)
            
            # If single segment exceeds chunk size, force split it
            if segment_tokens > self.chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split large segment by tokens
                chunks.extend(self._split_by_tokens(segment))
                continue
            
            # Check if adding segment would exceed chunk size
            if current_tokens + segment_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = [overlap_text, segment] if overlap_text else [segment]
                current_tokens = self.count_tokens(' '.join(current_chunk))
            else:
                # Add segment to current chunk
                current_chunk.append(segment)
                current_tokens += segment_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _get_overlap_text(self, chunks: List[str]) -> str:
        """
        Get overlap text from the end of previous chunks
        
        Args:
            chunks: List of text segments in current chunk
            
        Returns:
            Overlap text (last ~20% of current chunk)
        """
        if not chunks:
            return ""
        
        combined = ' '.join(chunks)
        tokens = self.count_tokens(combined)
        
        if tokens <= self.chunk_overlap:
            return combined
        
        # Take last chunk_overlap tokens worth of text
        words = combined.split()
        overlap_words = []
        overlap_tokens = 0
        
        for word in reversed(words):
            word_tokens = self.count_tokens(word)
            if overlap_tokens + word_tokens > self.chunk_overlap:
                break
            overlap_words.insert(0, word)
            overlap_tokens += word_tokens
        
        return ' '.join(overlap_words)
    
    def _split_by_tokens(self, text: str) -> List[str]:
        """
        Force split text by token count (last resort)
        
        Args:
            text: Text to split
            
        Returns:
            List of chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self.count_tokens(word)
            
            if current_tokens + word_tokens > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Add overlap
                overlap_words = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_words + [word]
                current_tokens = self.count_tokens(' '.join(current_chunk))
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a single document and create chunks
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of DocumentChunk objects
        """
        # Read file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
        
        # Clean LaTeX artifacts
        cleaned_text = self.clean_latex_artifacts(raw_text)
        
        # Get paper ID from filename
        paper_id = Path(file_path).stem
        
        # Split by sections
        sections = self.split_by_sections(cleaned_text)
        
        # Process each section
        all_chunks = []
        chunk_index = 0
        
        for section_name, section_text in sections:
            # Skip empty sections
            if not section_text.strip():
                continue
            
            # Recursively split section
            section_chunks = self.recursive_split(section_text, section_name)
            
            # Create DocumentChunk objects
            for chunk_text in section_chunks:
                chunk_text = chunk_text.strip()
                if len(chunk_text) < 50:  # Skip very small chunks
                    continue
                
                metadata = ChunkMetadata(
                    paper_id=paper_id,
                    chunk_index=chunk_index,
                    total_chunks=-1,  # Will update later
                    section=section_name,
                    char_count=len(chunk_text),
                    token_count=self.count_tokens(chunk_text),
                    source_file=Path(file_path).name
                )
                
                all_chunks.append(DocumentChunk(text=chunk_text, metadata=metadata))
                chunk_index += 1
        
        # Update total_chunks in metadata
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.metadata.total_chunks = total
        
        return all_chunks
    
    def process_directory(self, input_dir: str, output_file: str = "chunks.json") -> Dict:
        """
        Process all documents in a directory
        
        Args:
            input_dir: Directory containing .txt files
            output_file: Output JSON file path
            
        Returns:
            Statistics dictionary
        """
        input_path = Path(input_dir)
        txt_files = list(input_path.glob("*.txt"))
        
        print(f"Found {len(txt_files)} text files to process...")
        
        all_chunks = []
        stats = {
            'total_documents': len(txt_files),
            'total_chunks': 0,
            'total_tokens': 0,
            'avg_chunk_size': 0,
            'documents_processed': []
        }
        
        for i, file_path in enumerate(txt_files, 1):
            print(f"Processing {i}/{len(txt_files)}: {file_path.name}...")
            
            try:
                chunks = self.process_document(str(file_path))
                all_chunks.extend(chunks)
                
                doc_stats = {
                    'filename': file_path.name,
                    'chunks': len(chunks),
                    'tokens': sum(c.metadata.token_count for c in chunks)
                }
                stats['documents_processed'].append(doc_stats)
                
                print(f"  Created {len(chunks)} chunks")
                
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
                continue
        
        # Update statistics
        stats['total_chunks'] = len(all_chunks)
        stats['total_tokens'] = sum(c.metadata.token_count for c in all_chunks)
        stats['avg_chunk_size'] = stats['total_tokens'] / stats['total_chunks'] if stats['total_chunks'] > 0 else 0
        
        # Save to JSON
        output_data = {
            'chunks': [chunk.to_dict() for chunk in all_chunks],
            'statistics': stats,
            'config': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'encoding': 'cl100k_base'
            }
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Processing Complete!")
        print(f"{'='*60}")
        print(f"Total documents processed: {stats['total_documents']}")
        print(f"Total chunks created: {stats['total_chunks']}")
        print(f"Total tokens: {stats['total_tokens']:,}")
        print(f"Average chunk size: {stats['avg_chunk_size']:.1f} tokens")
        print(f"Output saved to: {output_path.absolute()}")
        print(f"{'='*60}\n")
        
        return stats


def main():
    """Main execution function"""
    # Configuration
    INPUT_DIR = "Input_Data"
    OUTPUT_FILE = "chunked_papers.json"
    CHUNK_SIZE = 512  # tokens
    CHUNK_OVERLAP = 100  # tokens (~20% overlap)
    
    # Create chunker
    chunker = AstrophysicsChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Process documents
    stats = chunker.process_directory(INPUT_DIR, OUTPUT_FILE)
    
    # Print some example chunks
    print("\nExample chunks from the output:")
    print("="*60)
    
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Show first 3 chunks
    for i, chunk in enumerate(data['chunks'][:3], 1):
        print(f"\nChunk {i}:")
        print(f"Paper ID: {chunk['metadata']['paper_id']}")
        print(f"Section: {chunk['metadata']['section']}")
        print(f"Tokens: {chunk['metadata']['token_count']}")
        print(f"Text preview: {chunk['text'][:200]}...")
        print("-"*60)


if __name__ == "__main__":
    main()
