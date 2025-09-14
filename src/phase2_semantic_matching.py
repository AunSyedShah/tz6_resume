"""
Phase 2: Enhanced Semantic Similarity Matching
Advanced transformer-based matching with contextual understanding
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedSemanticMatcher:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize advanced semantic matcher with specialized models
        """
        self.primary_model = SentenceTransformer(model_name)
        
        # Contextual patterns for different resume sections
        self.section_patterns = {
            'skills': [
                r'(technical\s+skills?|core\s+competencies|expertise|proficient\s+in|skilled\s+in)',
                r'(technologies?|programming|frameworks?|tools?|platforms?)'
            ],
            'experience': [
                r'(professional\s+experience|work\s+experience|employment|career)',
                r'(\d+\+?\s*years?\s+(?:of\s+)?experience|senior|lead|principal|architect)'
            ],
            'projects': [
                r'(projects?|implementations?|developed|built|created|designed)',
                r'(portfolio|achievements|accomplishments)'
            ],
            'education': [
                r'(education|degree|certification|training|course)',
                r'(university|college|institute|academy)'
            ]
        }
        
        # Skill context enhancers
        self.skill_contexts = {
            'python': ['data science', 'machine learning', 'web development', 'automation', 'django', 'flask'],
            'java': ['enterprise', 'spring', 'microservices', 'backend', 'android'],
            'javascript': ['frontend', 'react', 'angular', 'node.js', 'web development'],
            'aws': ['cloud', 'devops', 'infrastructure', 'scalability', 'ec2', 's3'],
            'business analysis': ['requirements', 'stakeholder', 'process improvement', 'documentation'],
            'machine learning': ['ai', 'data science', 'algorithms', 'predictive modeling', 'deep learning']
        }
    
    def extract_contextual_sections(self, text: str) -> Dict[str, str]:
        """
        Extract different sections from resume text for contextual analysis
        """
        sections = {
            'skills': '',
            'experience': '',
            'projects': '',
            'education': '',
            'full_text': text
        }
        
        text_lower = text.lower()
        
        for section, patterns in self.section_patterns.items():
            section_text = []
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    # Extract surrounding context (next 200 characters)
                    start = match.start()
                    end = min(start + 200, len(text))
                    context = text[start:end]
                    section_text.append(context)
            
            sections[section] = ' '.join(section_text)
        
        return sections
    
    def enhanced_similarity(self, text1: str, text2: str, 
                          focus_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Compute enhanced similarity with section-specific weighting
        """
        if focus_weights is None:
            focus_weights = {
                'skills': 0.4,
                'experience': 0.3,
                'projects': 0.2,
                'full_text': 0.1
            }
        
        # Extract sections from both texts
        sections1 = self.extract_contextual_sections(text1)
        sections2 = self.extract_contextual_sections(text2)
        
        similarities = {}
        weighted_score = 0.0
        
        for section in focus_weights.keys():
            if sections1[section] and sections2[section]:
                # Compute embeddings
                emb1 = self.primary_model.encode([sections1[section]])
                emb2 = self.primary_model.encode([sections2[section]])
                
                # Cosine similarity
                sim = cosine_similarity(emb1, emb2)[0][0]
                similarities[section] = float(sim)
                weighted_score += sim * focus_weights[section]
            else:
                similarities[section] = 0.0
        
        similarities['weighted_overall'] = weighted_score
        return similarities
    
    def contextual_skill_matching(self, jd_skills: List[str], 
                                resume_text: str) -> Dict[str, float]:
        """
        Match skills with contextual understanding
        """
        skill_scores = {}
        resume_lower = resume_text.lower()
        
        for skill in jd_skills:
            skill_lower = skill.lower()
            base_score = 0.0
            
            # Direct mention
            if skill_lower in resume_lower:
                base_score = 1.0
            else:
                # Semantic similarity with skill contexts
                if skill_lower in self.skill_contexts:
                    contexts = self.skill_contexts[skill_lower]
                    context_matches = sum(1 for ctx in contexts if ctx in resume_lower)
                    base_score = min(context_matches / len(contexts), 0.8)
            
            # Enhance with embedding similarity
            skill_embedding = self.primary_model.encode([skill])
            resume_embedding = self.primary_model.encode([resume_text[:500]])  # First 500 chars
            semantic_sim = cosine_similarity(skill_embedding, resume_embedding)[0][0]
            
            # Combined score
            final_score = max(base_score, semantic_sim * 0.7)
            skill_scores[skill] = float(final_score)
        
        return skill_scores
    
    def advanced_resume_ranking(self, job_description: str, 
                              resumes: List[Dict], 
                              top_n: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Advanced ranking with semantic understanding
        """
        jd_sections = self.extract_contextual_sections(job_description)
        
        # Extract key skills from JD
        jd_skills = self._extract_key_terms(job_description)
        
        resume_scores = []
        
        for resume in resumes:
            resume_text = resume.get('raw_text', '')
            if not resume_text:
                continue
            
            # Section-wise similarity
            similarities = self.enhanced_similarity(job_description, resume_text)
            
            # Contextual skill matching
            skill_matches = self.contextual_skill_matching(jd_skills, resume_text)
            
            # Experience level matching
            exp_score = self._match_experience_level(job_description, resume_text)
            
            # Combined scoring
            final_score = (
                similarities['weighted_overall'] * 0.5 +
                np.mean(list(skill_matches.values())) * 0.3 +
                exp_score * 0.2
            )
            
            # Detailed breakdown
            breakdown = {
                'semantic_similarity': similarities,
                'skill_matches': skill_matches,
                'experience_match': exp_score,
                'final_score': float(final_score)
            }
            
            resume_scores.append((resume['filename'], final_score, breakdown))
        
        # Sort by score
        resume_scores.sort(key=lambda x: x[1], reverse=True)
        return resume_scores[:top_n]
    
    def _extract_key_terms(self, text: str, max_terms: int = 10) -> List[str]:
        """
        Extract key terms from job description
        """
        # Simple keyword extraction (can be enhanced with TF-IDF or other methods)
        common_tech = [
            'python', 'java', 'javascript', 'react', 'angular', 'node.js', 'aws',
            'docker', 'kubernetes', 'machine learning', 'sql', 'mongodb',
            'business analysis', 'agile', 'scrum', 'project management'
        ]
        
        text_lower = text.lower()
        found_terms = [term for term in common_tech if term in text_lower]
        return found_terms[:max_terms]
    
    def _match_experience_level(self, jd_text: str, resume_text: str) -> float:
        """
        Match experience level requirements
        """
        # Extract experience requirements from JD
        jd_exp_pattern = r'(\d+)\+?\s*years?\s+(?:of\s+)?experience'
        jd_matches = re.findall(jd_exp_pattern, jd_text.lower())
        
        if not jd_matches:
            return 0.5  # Neutral if no experience specified
        
        required_years = int(jd_matches[0])
        
        # Extract experience from resume
        resume_exp_pattern = r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|exp)'
        resume_matches = re.findall(resume_exp_pattern, resume_text.lower())
        
        if not resume_matches:
            return 0.3  # Lower score if experience not clear
        
        candidate_years = max([int(match) for match in resume_matches])
        
        # Score based on experience match
        if candidate_years >= required_years:
            return 1.0
        elif candidate_years >= required_years * 0.7:
            return 0.8
        elif candidate_years >= required_years * 0.5:
            return 0.6
        else:
            return 0.3

# Test the advanced semantic matcher
if __name__ == "__main__":
    matcher = AdvancedSemanticMatcher()
    
    job_desc = """
    We are looking for a Senior Python Developer with 5+ years of experience.
    Must have expertise in machine learning, AWS cloud services, and Django framework.
    Experience with Docker and Kubernetes is preferred.
    """
    
    sample_resume = """
    Senior Software Engineer with 6 years of Python development experience.
    Expertise in machine learning algorithms and deep learning frameworks.
    Proficient in AWS services including EC2, S3, and Lambda.
    Experience with Django, Flask, and FastAPI frameworks.
    Skilled in Docker containerization and Kubernetes orchestration.
    """
    
    # Test enhanced similarity
    similarities = matcher.enhanced_similarity(job_desc, sample_resume)
    print("Enhanced Semantic Similarities:")
    for section, score in similarities.items():
        print(f"  {section}: {score:.3f}")
    
    # Test contextual skill matching
    jd_skills = ['python', 'machine learning', 'aws', 'django', 'docker']
    skill_matches = matcher.contextual_skill_matching(jd_skills, sample_resume)
    print(f"\nContextual Skill Matches:")
    for skill, score in skill_matches.items():
        print(f"  {skill}: {score:.3f}")