import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class EmbeddingGenerator:
    """
    A class to generate embeddings for text chunks using sentence-transformers.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the pre-trained model from sentence-transformers
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"Loaded model: {model_name}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the embedding
        """
        with torch.no_grad():
            embedding = self.model.encode(text, show_progress_bar=False)
        # Handle both Tensor and numpy array return types
        if isinstance(embedding, torch.Tensor):
            return embedding.cpu().numpy()
        return embedding  # Already a numpy array
    
    def process_chunks_directory(self, chunks_dir: str, output_dir: str):
        """
        Process all chunks in a directory structure and generate embeddings.
        For each chunk, loads url and title from the chunk's parent metadata (from chunking step).
        """
        # Create embeddings directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store all embeddings and metadata
        all_embeddings = {}
        
        # Try to load chunking metadata summary if available
        chunking_metadata_path = os.path.join(os.path.dirname(os.path.dirname(chunks_dir)), 'chunks_metadata.json')
        chunking_metadata = []
        if os.path.exists(chunking_metadata_path):
            with open(chunking_metadata_path, 'r', encoding='utf-8') as f:
                chunking_metadata = json.load(f)
        
        def get_url_title(chunk_file_path):
            # Extract category and chunk filename from the absolute path
            path_parts = chunk_file_path.replace('\\', '/').split('/')
            chunk_filename = path_parts[-1]  # e.g., "chunk_000.txt"
            category = path_parts[-2]  # e.g., "about-us-dcc-cares"
            
            # Create the expected chunk_file format from metadata
            expected_chunk_file = f"data/chunks\\{category}\\{chunk_filename}"
            expected_chunk_file_normalized = f"data/chunks/{category}/{chunk_filename}"
            
            # Search for matching entry in chunking metadata
            for m in chunking_metadata:
                metadata_chunk_file = m.get('chunk_file', '')
                # Normalize both paths for comparison
                metadata_normalized = metadata_chunk_file.replace('\\', '/')
                
                if (metadata_chunk_file == expected_chunk_file or 
                    metadata_normalized == expected_chunk_file_normalized):
                    return m.get('url'), m.get('title')
            
            return None, None
        
        # Walk through the chunks directory
        for root, dirs, files in os.walk(chunks_dir):
            if not files:
                continue
                
            # Get the relative path to determine the topic/category
            rel_path = os.path.relpath(root, chunks_dir)
            if rel_path == '.':
                continue
                
            print(f"Processing category: {rel_path}")
            
            # Process each chunk file in this category
            category_embeddings = []
            
            for file in tqdm(sorted(files), desc=f"Generating embeddings for {rel_path}"):
                if not file.endswith('.txt'):
                    continue
                    
                file_path = os.path.join(root, file)
                
                # Create a subdirectory in the embeddings folder
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                # Read the chunk content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    continue
                
                # Generate embedding
                embedding = self.generate_embedding(content)
                
                # Get url and title from chunking metadata
                url, title = get_url_title(file_path)
                
                # Create metadata
                chunk_data = {
                    'chunk_id': os.path.splitext(file)[0],
                    'category': rel_path,
                    'file_path': file_path,
                    'content': content,
                    'url': url,
                    'title': title,
                    'embedding': embedding.tolist()  # Convert to list for JSON serialization
                }
                
                category_embeddings.append(chunk_data)
                
                # Save individual embedding
                output_file = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    # Save without the embedding to keep the file size manageable
                    save_data = {**chunk_data}
                    # Remove the embedding from the individual file to save space
                    # We'll keep them in the full embeddings file
                    del save_data['embedding']
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            all_embeddings[rel_path] = category_embeddings
        
        # Save all embeddings to a single file
        print("Saving all embeddings to a single file...")
        embeddings_file = os.path.join(output_dir, 'all_embeddings.json')
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(all_embeddings, f, ensure_ascii=False, indent=2)
        
        # Also save embeddings in a format suitable for FAISS
        print("Preparing data for FAISS...")
        faiss_embeddings = []
        faiss_metadata = []
        
        for category, embeddings in all_embeddings.items():
            for item in embeddings:
                faiss_embeddings.append(item['embedding'])
                # Create metadata without the embedding
                meta = {
                    'chunk_id': item['chunk_id'],
                    'category': item['category'],
                    'file_path': item['file_path'],
                    'content': item['content'],
                    'url': item.get('url'),
                    'title': item.get('title')
                }
                faiss_metadata.append(meta)
        
        # Save FAISS-ready data
        faiss_dir = os.path.join(output_dir, 'faiss')
        os.makedirs(faiss_dir, exist_ok=True)
        
        # Save embeddings as numpy array
        embeddings_array = np.array(faiss_embeddings, dtype=np.float32)
        np.save(os.path.join(faiss_dir, 'embeddings.npy'), embeddings_array)
        
        # Save metadata
        with open(os.path.join(faiss_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(faiss_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully processed {len(faiss_metadata)} chunks.")
        print(f"Embeddings saved to {output_dir}")
        print(f"FAISS-ready data saved to {faiss_dir}")


def main():
    # Configuration
    chunks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'chunks')
    embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'embeddings')
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()
    
    # Process all chunks
    embedding_generator.process_chunks_directory(chunks_dir, embeddings_dir)


if __name__ == "__main__":
    main()
