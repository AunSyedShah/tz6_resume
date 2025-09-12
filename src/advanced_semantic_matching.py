"""
Advanced Semantic Matching Engine with FAISS Vector Database
Uses state-of-the-art embedding models and vector similarity search
"""

import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import json
from config import DATA_DIR

class AdvancedSemanticMatcher:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize with advanced embedding model
        all-mpnet-base-v2 is currently one of the best sentence transformers
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # FAISS index for vector similarity search
        self.faiss_index = None
        self.resume_embeddings = None
        self.resume_metadata = None
        
        # Cache paths
        self.cache_dir = os.path.join(DATA_DIR, "embeddings_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.index_path = os.path.join(self.cache_dir, f"faiss_index_{model_name.replace('/', '_')}.index")
        self.embeddings_path = os.path.join(self.cache_dir, f"embeddings_{model_name.replace('/', '_')}.pkl")
        self.metadata_path = os.path.join(self.cache_dir, f"metadata_{model_name.replace('/', '_')}.pkl")
    
    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for list of texts
        """
        return self.model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)
    
    def build_faiss_index(self, resume_data: List[Dict], force_rebuild: bool = False) -> None:
        """
        Build FAISS index from resume data with caching
        """
        if not force_rebuild and os.path.exists(self.index_path):
            print("Loading cached FAISS index...")
            self.load_cached_index()
            return
        
        print(f"Building FAISS index with {len(resume_data)} resumes...")
        
        # Prepare texts for embedding
        resume_texts = []
        metadata = []
        
        for resume in resume_data:
            # Combine multiple text fields for richer embeddings
            combined_text = self._combine_resume_text(resume)
            resume_texts.append(combined_text)
            
            # Store metadata for retrieval
            metadata.append({
                'filename': resume['filename'],
                'skills': resume.get('features', {}).get('skills', []),
                'roles': resume.get('features', {}).get('roles', []),
                'education': resume.get('features', {}).get('education', []),
                'experience_years': resume.get('features', {}).get('experience_years', 0),
                'raw_text': resume.get('raw_text', '')
            })
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.encode_texts(resume_texts)
        
        # Build FAISS index
        print("Building FAISS index...")
        self.faiss_index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings.astype('float32'))
        
        self.resume_embeddings = embeddings
        self.resume_metadata = metadata
        
        # Cache the index and data
        self.save_cache()
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def _combine_resume_text(self, resume: Dict) -> str:
        """
        Combine multiple resume fields into rich text for embedding
        """
        parts = []
        
        # Raw text (most important)
        if 'raw_text' in resume:
            parts.append(resume['raw_text'])
        
        # Features
        if 'features' in resume:
            features = resume['features']
            
            # Skills with emphasis
            if 'skills' in features and features['skills']:
                skills_text = "Key skills: " + ", ".join(features['skills'])
                parts.append(skills_text)
            
            # Roles with emphasis
            if 'roles' in features and features['roles']:
                roles_text = "Job roles: " + ", ".join(features['roles'])
                parts.append(roles_text)
            
            # Education
            if 'education' in features and features['education']:
                edu_text = "Education: " + ", ".join(features['education'])
                parts.append(edu_text)
            
            # Experience
            if 'experience_years' in features and features['experience_years']:
                exp_text = f"Experience: {features['experience_years']} years"
                parts.append(exp_text)
        
        return " ".join(parts)
    
    def save_cache(self) -> None:
        """Save FAISS index and metadata to disk"""
        faiss.write_index(self.faiss_index, self.index_path)
        
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.resume_embeddings, f)
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.resume_metadata, f)
    
    def load_cached_index(self) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            self.faiss_index = faiss.read_index(self.index_path)
            
            with open(self.embeddings_path, 'rb') as f:
                self.resume_embeddings = pickle.load(f)
            
            with open(self.metadata_path, 'rb') as f:
                self.resume_metadata = pickle.load(f)
            
            print(f"Loaded cached index with {len(self.resume_metadata)} resumes")
            return True
        except Exception as e:
            print(f"Failed to load cached index: {e}")
            return False
    
    def semantic_search(self, query: str, top_k: int = 50) -> List[Tuple[Dict, float]]:
        """
        Perform semantic search using FAISS
        Returns list of (metadata, similarity_score) tuples
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        similarities, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx != -1:  # Valid result
                results.append((self.resume_metadata[idx], float(similarity)))
        
        return results
    
    def advanced_score(self, resume_text: str, jd_text: str, resume_metadata: Dict = None) -> Dict[str, float]:
        """
        Calculate advanced semantic similarity scores
        Returns multiple similarity metrics
        """
        # Basic semantic similarity
        embeddings = self.encode_texts([resume_text, jd_text], show_progress=False)
        faiss.normalize_L2(embeddings)
        
        basic_similarity = float(np.dot(embeddings[0], embeddings[1]))
        
        # Skills-focused similarity (if metadata available)
        skills_similarity = 0.0
        if resume_metadata and 'skills' in resume_metadata:
            resume_skills = resume_metadata['skills']
            if resume_skills:
                skills_text = "Skills: " + ", ".join(resume_skills)
                skills_embeddings = self.encode_texts([skills_text, jd_text], show_progress=False)
                faiss.normalize_L2(skills_embeddings)
                skills_similarity = float(np.dot(skills_embeddings[0], skills_embeddings[1]))
        
        # Role-focused similarity
        role_similarity = 0.0
        if resume_metadata and 'roles' in resume_metadata:
            resume_roles = resume_metadata['roles']
            if resume_roles:
                roles_text = "Roles: " + ", ".join(resume_roles)
                role_embeddings = self.encode_texts([roles_text, jd_text], show_progress=False)
                faiss.normalize_L2(role_embeddings)
                role_similarity = float(np.dot(role_embeddings[0], role_embeddings[1]))
        
        # Combined advanced score
        weights = {
            'basic': 0.4,
            'skills': 0.35,
            'roles': 0.25
        }
        
        advanced_score = (
            weights['basic'] * basic_similarity +
            weights['skills'] * skills_similarity +
            weights['roles'] * role_similarity
        )
        
        return {
            'advanced_score': advanced_score,
            'basic_similarity': basic_similarity,
            'skills_similarity': skills_similarity,
            'role_similarity': role_similarity
        }
    
    def batch_score_resumes(self, resumes_data: List[Dict], jd_text: str) -> List[Dict]:
        """
        Score all resumes against JD using advanced semantic matching
        """
        if self.faiss_index is None:
            print("Building FAISS index for batch scoring...")
            self.build_faiss_index(resumes_data)
        
        results = []
        
        print(f"Scoring {len(resumes_data)} resumes with advanced semantic matching...")
        
        for i, resume in enumerate(resumes_data):
            if i % 50 == 0:  # Progress update
                print(f"Progress: {i}/{len(resumes_data)} resumes processed")
            
            resume_text = self._combine_resume_text(resume)
            resume_metadata = {
                'skills': resume.get('features', {}).get('skills', []),
                'roles': resume.get('features', {}).get('roles', []),
                'education': resume.get('features', {}).get('education', []),
                'experience_years': resume.get('features', {}).get('experience_years', 0)
            }
            
            scores = self.advanced_score(resume_text, jd_text, resume_metadata)
            
            result = {
                'filename': resume['filename'],
                'advanced_scores': scores,
                'metadata': resume_metadata
            }
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'description': 'Advanced sentence transformer with FAISS vector search'
        }

# Global instance for reuse
_global_matcher = None

def get_advanced_matcher(model_name: str = "all-mpnet-base-v2") -> AdvancedSemanticMatcher:
    """Get global instance of advanced matcher"""
    global _global_matcher
    if _global_matcher is None or _global_matcher.model_name != model_name:
        _global_matcher = AdvancedSemanticMatcher(model_name)
    return _global_matcher

if __name__ == "__main__":
    # Test the advanced semantic matcher
    matcher = AdvancedSemanticMatcher()
    
    # Test basic functionality
    texts = ["Python developer with machine learning experience", "Java backend engineer"]
    embeddings = matcher.encode_texts(texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Test similarity
    scores = matcher.advanced_score(texts[0], "Looking for Python ML engineer", 
                                  {'skills': ['python', 'machine learning'], 'roles': ['developer']})
    print(f"Advanced scores: {scores}")
