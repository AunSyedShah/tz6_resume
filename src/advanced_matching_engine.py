import sys
sys.path.append('/workspaces/tz6_resume/src')

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json
import pickle
import os
from preprocessing import preprocess_text
from nlp_features import extract_skills, advanced_nlp_features
from semantic_matching import embedding_engine, compute_semantic_similarity
from config import DATA_DIR

class AdvancedMatchingEngine:
    """
    Advanced matching engine with ML-based scoring and FAISS integration
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,
            max_df=0.8
        )
        self.ml_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Cache directory
        self.cache_dir = os.path.join(DATA_DIR, 'model_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def extract_advanced_features(self, resume_data, jd_data):
        """
        Extract comprehensive features for ML model
        """
        features = []
        
        # 1. TF-IDF similarity
        resume_text = resume_data['processed_text']
        jd_text = jd_data['processed_text']
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([resume_text, jd_text])
            tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_sim = 0.0
        
        features.append(tfidf_sim)
        
        # 2. Semantic similarity
        semantic_sim = compute_semantic_similarity(resume_text, jd_text)
        features.append(semantic_sim)
        
        # 3. Skills overlap features
        resume_skills = set(resume_data['features'].get('skills', []))
        jd_skills = set(jd_data['features'].get('skills', []))
        
        if jd_skills:
            skill_overlap = len(resume_skills & jd_skills) / len(jd_skills)
            skill_coverage = len(resume_skills & jd_skills) / len(resume_skills) if resume_skills else 0
        else:
            skill_overlap = skill_coverage = 0
        
        features.extend([skill_overlap, skill_coverage])
        
        # 4. Role matching
        resume_roles = set([r.lower().strip() for r in resume_data['features'].get('roles', [])])
        jd_roles = set([r.lower().strip() for r in jd_data['features'].get('roles', [])])
        
        role_match = 1.0 if resume_roles & jd_roles else 0.0
        features.append(role_match)
        
        # 5. Education level matching
        resume_edu = set([e.lower() for e in resume_data['features'].get('education', [])])
        jd_edu = set([e.lower() for e in jd_data['features'].get('education', [])])
        
        edu_match = 1.0 if resume_edu & jd_edu else 0.0
        features.append(edu_match)
        
        # 6. Experience features
        resume_exp = resume_data['features'].get('experience_years', 0)
        jd_exp = jd_data['features'].get('experience_years', 0)
        
        exp_diff = abs(resume_exp - jd_exp) if jd_exp > 0 else 0
        exp_ratio = min(resume_exp / jd_exp, 2.0) if jd_exp > 0 else 1.0
        
        features.extend([resume_exp, exp_diff, exp_ratio])
        
        # 7. Text length features
        resume_len = len(resume_text.split())
        jd_len = len(jd_text.split())
        len_ratio = resume_len / jd_len if jd_len > 0 else 1.0
        
        features.extend([resume_len, jd_len, len_ratio])
        
        # 8. Advanced skill categorization overlap
        resume_categorized = resume_data['features'].get('categorized_skills', {})
        jd_categorized = jd_data['features'].get('categorized_skills', {})
        
        category_matches = []
        for category in ['Programming Languages', 'Web Technologies', 'Databases', 'ML/AI', 'Cloud', 'Tools']:
            resume_cat = set(resume_categorized.get(category, []))
            jd_cat = set(jd_categorized.get(category, []))
            
            if jd_cat:
                cat_overlap = len(resume_cat & jd_cat) / len(jd_cat)
            else:
                cat_overlap = 0.0
            
            category_matches.append(cat_overlap)
        
        features.extend(category_matches)
        
        return np.array(features)
    
    def train_model(self, training_data):
        """
        Train ML model with historical hiring decisions
        """
        print("Training advanced matching model...")
        
        X = []
        y = []
        
        for data_point in training_data:
            features = self.extract_advanced_features(
                data_point['resume_data'], 
                data_point['jd_data']
            )
            X.append(features)
            y.append(data_point['score'])  # Historical hiring score
        
        X = np.array(X)
        y = np.array(y)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.ml_model.fit(X_scaled, y)
        self.is_trained = True
        
        # Cache trained model
        model_file = os.path.join(self.cache_dir, 'matching_model.pkl')
        scaler_file = os.path.join(self.cache_dir, 'scaler.pkl')
        
        with open(model_file, 'wb') as f:
            pickle.dump(self.ml_model, f)
        
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model trained and cached. Training accuracy: {self.ml_model.score(X_scaled, y):.3f}")
    
    def load_trained_model(self):
        """
        Load pre-trained model from cache
        """
        model_file = os.path.join(self.cache_dir, 'matching_model.pkl')
        scaler_file = os.path.join(self.cache_dir, 'scaler.pkl')
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            with open(model_file, 'rb') as f:
                self.ml_model = pickle.load(f)
            
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_trained = True
            print("Loaded pre-trained matching model from cache")
            return True
        
        return False
    
    def advanced_score(self, resume_data, jd_data):
        """
        Compute advanced matching score using ML model
        """
        # Extract features
        features = self.extract_advanced_features(resume_data, jd_data)
        
        # If model is trained, use ML prediction
        if self.is_trained:
            features_scaled = self.scaler.transform([features])
            ml_score = self.ml_model.predict(features_scaled)[0]
            ml_score = np.clip(ml_score, 0.0, 1.0)  # Ensure score is between 0 and 1
        else:
            ml_score = 0.0
        
        # Fallback: Rule-based scoring
        tfidf_sim = features[0]
        semantic_sim = features[1]
        skill_overlap = features[2]
        role_match = features[4]
        edu_match = features[5]
        
        # Weighted combination for rule-based score
        rule_based_score = (
            0.25 * tfidf_sim +
            0.30 * semantic_sim +
            0.25 * skill_overlap +
            0.15 * role_match +
            0.05 * edu_match
        )
        
        # Combine ML and rule-based scores
        if self.is_trained:
            final_score = 0.7 * ml_score + 0.3 * rule_based_score
        else:
            final_score = rule_based_score
        
        return {
            'final_score': float(final_score),
            'ml_score': float(ml_score),
            'rule_based_score': float(rule_based_score),
            'feature_breakdown': {
                'tfidf_similarity': float(tfidf_sim),
                'semantic_similarity': float(semantic_sim),
                'skill_overlap': float(skill_overlap),
                'role_match': float(role_match),
                'education_match': float(edu_match)
            }
        }
    
    def batch_score(self, resumes_data, jd_data):
        """
        Score multiple resumes efficiently
        """
        results = []
        
        for resume_data in resumes_data:
            score_result = self.advanced_score(resume_data, jd_data)
            results.append({
                'filename': resume_data.get('filename', ''),
                'score': score_result['final_score'],
                'detailed_scores': score_result
            })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def hybrid_search(self, jd_data, resumes_data, use_faiss=True, top_k=20):
        """
        Hybrid search combining FAISS semantic search and advanced scoring
        """
        if use_faiss and embedding_engine.faiss_index is not None:
            # Step 1: Use FAISS for initial candidate retrieval
            query_text = f"{jd_data['raw_text']} {' '.join(jd_data['features'].get('skills', []))}"
            faiss_results = embedding_engine.semantic_search(query_text, top_k=min(top_k * 2, len(resumes_data)))
            
            # Step 2: Get corresponding resume data
            candidate_resumes = []
            for result in faiss_results:
                filename = result['metadata']['filename']
                resume_data = next((r for r in resumes_data if r.get('filename') == filename), None)
                if resume_data:
                    candidate_resumes.append(resume_data)
            
        else:
            # Use all resumes if FAISS not available
            candidate_resumes = resumes_data
        
        # Step 3: Advanced scoring on candidates
        final_results = self.batch_score(candidate_resumes, jd_data)
        
        return final_results[:top_k]

# Global advanced matching engine
advanced_engine = AdvancedMatchingEngine()

def combined_score(resume_data, jd_data):
    """
    Backward compatibility function
    """
    result = advanced_engine.advanced_score(resume_data, jd_data)
    return result['final_score']

def advanced_rank_resumes(jd_data, resumes_data, top_n=10, use_ml=True):
    """
    Advanced resume ranking with ML and FAISS
    """
    if use_ml and not advanced_engine.is_trained:
        # Try to load pre-trained model
        advanced_engine.load_trained_model()
    
    return advanced_engine.hybrid_search(jd_data, resumes_data, top_k=top_n)

if __name__ == "__main__":
    # Test advanced matching engine
    print("Testing Advanced Matching Engine...")
    
    # Load sample data
    try:
        with open(f"{DATA_DIR}/processed/resumes_processed.json", 'r') as f:
            resumes = json.load(f)[:10]  # Test with first 10
    except:
        print("Sample resume data not found")
        resumes = []
    
    if resumes:
        # Sample JD
        sample_jd = {
            'raw_text': "Senior Python Developer with machine learning experience",
            'processed_text': preprocess_text("Senior Python Developer with machine learning experience"),
            'features': advanced_nlp_features("Senior Python Developer with machine learning experience")
        }
        
        # Test scoring
        for resume in resumes[:3]:
            score_result = advanced_engine.advanced_score(resume, sample_jd)
            print(f"\nResume: {resume.get('filename', 'Unknown')}")
            print(f"Final Score: {score_result['final_score']:.3f}")
            print(f"Feature Breakdown: {score_result['feature_breakdown']}")
