import os
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
import json

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

class TextChunker:
    """
    A class to handle text chunking with overlap and proper sentence boundaries.
    """
    
    def __init__(self, chunk_size: int = 200, overlap_size: int = 50):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum number of words per chunk
            overlap_size: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
    
    def chunk_text_with_overlap(self, text: str) -> list:
        """
        Chunk text into overlapping segments based on sentences.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            print(f"Warning: NLTK tokenization failed, using simple split: {e}")
            # Fallback to simple sentence splitting
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        if not sentences:
            return [text] if text.strip() else []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = len(sentence.split())
            
            # If adding this sentence would exceed max_tokens and we have content
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_chunk = []
                overlap_count = 0
                
                # Go backwards from current position to build overlap
                j = len(current_chunk) - 1
                while j >= 0 and overlap_count < self.overlap_size:
                    sentence_tokens_overlap = len(current_chunk[j].split())
                    if overlap_count + sentence_tokens_overlap <= self.overlap_size:
                        overlap_chunk.insert(0, current_chunk[j])
                        overlap_count += sentence_tokens_overlap
                        j -= 1
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_tokens = overlap_count
            
            # Handle sentences that are longer than max_tokens
            if sentence_tokens > self.chunk_size:
                # If we have existing content, save it first
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if chunk_text.strip():
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence by words
                words = sentence.split()
                for k in range(0, len(words), self.chunk_size):
                    word_chunk = " ".join(words[k:k + self.chunk_size])
                    if word_chunk.strip():
                        chunks.append(word_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            
            i += 1
        
        # Add final chunk if it exists
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def process_files(self, input_dir: str, output_dir: str) -> tuple:
        """
        Process all text files in the input directory and create chunks.
        Adds url and title from corresponding .json file to each chunk's metadata.
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)
        documents = []
        metadata = []
        txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        if not txt_files:
            print(f"No .txt files found in {input_dir}")
            return documents, metadata
        print(f"Processing {len(txt_files)} files from {input_dir}...")
        for filename in txt_files:
            file_path = os.path.join(input_dir, filename)
            slug = filename[:-4]
            # Try to load url/title from .json
            url = None
            title = None
            json_path = os.path.join(input_dir, f"{slug}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as jf:
                        jdata = json.load(jf)
                        url = jdata.get("url")
                        title = jdata.get("title")
                except Exception as e:
                    print(f"  Warning: Could not read {json_path}: {e}")
            print(f"Processing {filename}...")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                if not raw_text.strip():
                    print(f"  Warning: {filename} is empty, skipping...")
                    continue
                chunks = self.chunk_text_with_overlap(raw_text)
                if not chunks:
                    print(f"  Warning: No chunks created for {filename}")
                    continue
                file_chunks_dir = os.path.join(output_dir, filename.replace('.txt', ''))
                os.makedirs(file_chunks_dir, exist_ok=True)
                for i, chunk in enumerate(chunks):
                    chunk_filename = f"chunk_{i:03d}.txt"
                    chunk_path = os.path.join(file_chunks_dir, chunk_filename)
                    try:
                        with open(chunk_path, "w", encoding="utf-8") as chunk_file:
                            chunk_file.write(chunk)
                        documents.append(chunk)
                        metadata.append({
                            "source": filename,
                            "chunk_id": i,
                            "chunk_file": chunk_path,
                            "word_count": len(chunk.split()),
                            "char_count": len(chunk),
                            "url": url,
                            "title": title
                        })
                    except Exception as e:
                        print(f"  Error saving chunk {i} for {filename}: {e}")
                        continue
                print(f"  Created {len(chunks)} chunks for {filename}")
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                continue
        return documents, metadata

def main():
    """Main function to run the chunking process."""
    
    # Parameters
    input_folder = "data/raw"
    output_folder = "data/chunks"
    chunk_size = 200  # number of words per chunk
    overlap_size = 50  # number of words to overlap between chunks
    
    # Initialize chunker
    chunker = TextChunker(chunk_size=chunk_size, overlap_size=overlap_size)
    
    try:
        # Process files and create chunks
        documents, metadata = chunker.process_files(input_folder, output_folder)
        
        if not documents:
            print("No documents were processed successfully.")
            return
        
        print(f"\nSUMMARY:")
        print(f"Total chunks created: {len(documents)}")
        print(f"Source files processed: {len(set(m['source'] for m in metadata))}")
        print(f"Chunks saved to: {output_folder}")
        
        # Save metadata summary
        metadata_path = "chunks_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Metadata saved to: {metadata_path}")
        
        # Display statistics
        word_counts = [m['word_count'] for m in metadata]
        print(f"\nChunk Statistics:")
        print(f"  Average words per chunk: {np.mean(word_counts):.1f}")
        print(f"  Min words per chunk: {min(word_counts)}")
        print(f"  Max words per chunk: {max(word_counts)}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()


