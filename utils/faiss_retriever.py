import numpy as np
import faiss
import json
from pathlib import Path
from typing import List, Dict, Optional

class FaissRetriever:
    def __init__(self, embeddings_dir: str = 'data/embeddings/faiss'):
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings = None
        self.metadata = None
        self.index = None
        self._load_index()

    def _load_index(self):
        embeddings_path = self.embeddings_dir / 'embeddings.npy'
        metadata_path = self.embeddings_dir / 'metadata.json'
        self.embeddings = np.load(embeddings_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        if self.index is None or self.metadata is None:
            raise ValueError("FAISS index or metadata not loaded.")
        D, I = self.index.search(query_embedding.astype(np.float32), top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx]
            meta['distance'] = float(dist)
            results.append(meta)
        return results 