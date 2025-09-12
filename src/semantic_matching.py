import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from config import DATA_DIR

class AdvancedEmbeddingEngine:
    """
    Advanced embedding engine with FAISS vector database support
    """
    
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize with advanced embedding model
        """
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # FAISS index for fast similarity search
        self.faiss_index = None
        self.resume_texts = []
        self.resume_metadata = []
        
        # Cache directory
        self.cache_dir = os.path.join(DATA_DIR, 'embeddings_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"Initialized embedding engine with model: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def encode_text(self, text, use_gpu=True):
        """
        Advanced text encoding with GPU support
        """
        if torch.cuda.is_available() and use_gpu:
            device = 'cuda'
        else:
            device = 'cpu'
            
        # Move model to device
        if hasattr(self.embedding_model, 'to'):
            self.embedding_model = self.embedding_model.to(device)
        
        # Generate embeddings with normalization
        embeddings = self.embedding_model.encode(
            text, 
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embeddings.cpu().numpy() if torch.cuda.is_available() else embeddings
    
    def build_faiss_index(self, resume_data_list):
        """
        Build FAISS index for fast similarity search
        """
        print("Building FAISS index for resume embeddings...")
        
        # Check if cached embeddings exist
        cache_file = os.path.join(self.cache_dir, 'resume_embeddings.pkl')
        index_file = os.path.join(self.cache_dir, 'faiss_index.bin')
        
        if os.path.exists(cache_file) and os.path.exists(index_file):
            print("Loading cached embeddings and FAISS index...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.resume_texts = cache_data['texts']
                self.resume_metadata = cache_data['metadata']
                embeddings = cache_data['embeddings']
            
            self.faiss_index = faiss.read_index(index_file)
            print(f"Loaded {len(self.resume_texts)} cached embeddings")
            return
        
        # Generate new embeddings
        texts = []
        metadata = []
        
        for resume_data in resume_data_list:
            # Combine different text fields for better representation
            combined_text = f"""
            {resume_data.get('raw_text', '')}
            Skills: {' '.join(resume_data.get('features', {}).get('skills', []))}
            Roles: {' '.join(resume_data.get('features', {}).get('roles', []))}
            Education: {' '.join(resume_data.get('features', {}).get('education', []))}
            """.strip()
            
            texts.append(combined_text)
            metadata.append({
                'filename': resume_data.get('filename', ''),
                'skills': resume_data.get('features', {}).get('skills', []),
                'roles': resume_data.get('features', {}).get('roles', []),
                'education': resume_data.get('features', {}).get('education', []),
                'experience': resume_data.get('features', {}).get('experience_years', 0)
            })
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode_text(batch_texts)
            all_embeddings.append(batch_embeddings)
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} embeddings")
        
        embeddings = np.vstack(all_embeddings)
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for normalized vectors
        self.faiss_index.add(embeddings.astype('float32'))
        
        self.resume_texts = texts
        self.resume_metadata = metadata
        
        # Cache embeddings and index
        cache_data = {
            'texts': texts,
            'metadata': metadata,
            'embeddings': embeddings
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        faiss.write_index(self.faiss_index, index_file)
        
        print(f"Built FAISS index with {len(texts)} resume embeddings")
    
    def semantic_search(self, query_text, top_k=10):
        """
        Perform semantic search using FAISS
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index() first.")
        
        # Encode query
        query_embedding = self.encode_text([query_text])
        
        # Search in FAISS
        similarities, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.resume_metadata):
                results.append({
                    'rank': i + 1,
                    'similarity': float(similarity),
                    'metadata': self.resume_metadata[idx],
                    'text_snippet': self.resume_texts[idx][:200] + "..."
                })
        
        return results
    
    def compute_advanced_similarity(self, text1, text2):
        """
        Compute advanced semantic similarity with multiple techniques
        """
        # Method 1: Direct similarity using advanced model
        embeddings = self.encode_text([text1, text2])
        cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Method 2: Weighted similarity based on text length and content
        text1_words = set(text1.lower().split())
        text2_words = set(text2.lower().split())
        
        # Jaccard similarity
        intersection = len(text1_words.intersection(text2_words))
        union = len(text1_words.union(text2_words))
        jaccard_sim = intersection / union if union > 0 else 0
        
        # Combined similarity (weighted)
        combined_sim = 0.8 * cosine_sim + 0.2 * jaccard_sim
        
        return {
            'cosine_similarity': float(cosine_sim),
            'jaccard_similarity': float(jaccard_sim),
            'combined_similarity': float(combined_sim)
        }

# Global embedding engine instance
embedding_engine = AdvancedEmbeddingEngine()

def compute_semantic_similarity(text1, text2):
    """
    Backward compatibility function
    """
    result = embedding_engine.compute_advanced_similarity(text1, text2)
    return result['combined_similarity']

def initialize_vector_db(resume_data_list):
    """
    Initialize FAISS vector database with resume data
    """
    embedding_engine.build_faiss_index(resume_data_list)

def semantic_search_resumes(query_text, top_k=10):
    """
    Search resumes using semantic similarity
    """
    return embedding_engine.semantic_search(query_text, top_k)

if __name__ == "__main__":
    # Test advanced semantic matching
    text1 = "Senior Python Developer with machine learning and AI experience"
    text2 = "Data Scientist skilled in Python, TensorFlow, and deep learning"
    
    result = embedding_engine.compute_advanced_similarity(text1, text2)
    print("Advanced Semantic Similarity Results:")
    print(f"Cosine Similarity: {result['cosine_similarity']:.3f}")
    print(f"Jaccard Similarity: {result['jaccard_similarity']:.3f}")
    print(f"Combined Similarity: {result['combined_similarity']:.3f}")
    
    # Test simple interface
    simple_sim = compute_semantic_similarity(text1, text2)
    print(f"Simple Interface Result: {simple_sim:.3f}")
