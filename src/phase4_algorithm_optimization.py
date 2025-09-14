"""
Phase 4: Algorithm Fine-tuning and Optimization
==============================================

This module implements advanced ranking algorithms, ensemble methods,
and machine learning optimizations to maximize matching accuracy.

Key Features:
- Ensemble scoring with multiple algorithms
- Advanced weighted ranking systems
- Machine learning model optimization
- Contextual boost algorithms
- Performance tuning and calibration
- Multi-dimensional similarity scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import previous phase components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_skill_mapping import SkillSynonymMapper, EnhancedSkillExtractor
from phase2_semantic_matching import AdvancedSemanticMatcher
from phase3_industry_terminology import EnhancedIndustryMatcher

logger = logging.getLogger(__name__)


@dataclass
class MatchingFeatures:
    """Comprehensive feature set for resume-job matching."""
    skill_match: float = 0.0
    experience_match: float = 0.0
    seniority_match: float = 0.0
    domain_relevance: float = 0.0
    technical_depth: float = 0.0
    certification_match: float = 0.0
    education_match: float = 0.0
    keyword_density: float = 0.0
    role_alignment: float = 0.0
    industry_experience: float = 0.0
    leadership_indicators: float = 0.0
    technology_freshness: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for ML processing."""
        return {
            'skill_match': self.skill_match,
            'experience_match': self.experience_match,
            'seniority_match': self.seniority_match,
            'domain_relevance': self.domain_relevance,
            'technical_depth': self.technical_depth,
            'certification_match': self.certification_match,
            'education_match': self.education_match,
            'keyword_density': self.keyword_density,
            'role_alignment': self.role_alignment,
            'industry_experience': self.industry_experience,
            'leadership_indicators': self.leadership_indicators,
            'technology_freshness': self.technology_freshness
        }
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML processing."""
        return np.array(list(self.to_dict().values()))


class EnsembleMatchingEngine:
    """Advanced ensemble matching engine with multiple algorithms."""
    
    def __init__(self):
        # Initialize component matchers
        self.skill_mapper = SkillSynonymMapper()
        self.skill_extractor = EnhancedSkillExtractor()
        self.semantic_matcher = AdvancedSemanticMatcher()
        self.industry_matcher = EnhancedIndustryMatcher()
        
        # Initialize ML models
        self.models = self._initialize_models()
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Ensemble weights (can be learned from data)
        self.ensemble_weights = {
            'skill_focused': 0.3,      # Traditional skill matching
            'semantic_focused': 0.25,   # Semantic understanding
            'industry_focused': 0.2,    # Domain expertise
            'experience_focused': 0.15, # Experience alignment
            'ml_optimized': 0.1        # ML-learned patterns
        }
        
        # Feature importance weights
        self.feature_weights = self._initialize_feature_weights()
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize ensemble ML models."""
        return {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'linear_regression': LinearRegression()
        }
    
    def _initialize_feature_weights(self) -> Dict[str, float]:
        """Initialize feature importance weights."""
        return {
            'skill_match': 0.25,
            'experience_match': 0.15,
            'seniority_match': 0.12,
            'domain_relevance': 0.15,
            'technical_depth': 0.10,
            'certification_match': 0.08,
            'education_match': 0.06,
            'keyword_density': 0.04,
            'role_alignment': 0.03,
            'industry_experience': 0.02
        }
    
    def extract_comprehensive_features(self, jd_text: str, resume_text: str) -> MatchingFeatures:
        """Extract comprehensive feature set for matching."""
        features = MatchingFeatures()
        
        # Phase 1: Enhanced skill matching
        jd_skills = self.skill_extractor.extract_enhanced_skills(jd_text)
        resume_skills = self.skill_extractor.extract_enhanced_skills(resume_text)
        features.skill_match = self._calculate_skill_similarity(jd_skills, resume_skills)
        
        # Phase 2: Semantic matching  
        features.experience_match = self.semantic_matcher._match_experience_level(jd_text, resume_text)
        # For seniority and technical depth, use simple pattern matching for now
        features.seniority_match = self._match_seniority_level(jd_text, resume_text)
        features.technical_depth = self._assess_technical_complexity(jd_text, resume_text)
        
        # Phase 3: Industry matching
        industry_result = self.industry_matcher.enhanced_skill_match(jd_text, resume_text)
        features.domain_relevance = industry_result.get('domain_relevance', 0.0)
        
        # Advanced features
        features.certification_match = self._calculate_certification_match(jd_text, resume_text)
        features.education_match = self._calculate_education_match(jd_text, resume_text)
        features.keyword_density = self._calculate_keyword_density(jd_text, resume_text)
        features.role_alignment = self._calculate_role_alignment(jd_text, resume_text)
        features.industry_experience = self._calculate_industry_experience(jd_text, resume_text)
        features.leadership_indicators = self._detect_leadership_indicators(resume_text)
        features.technology_freshness = self._assess_technology_freshness(resume_text)
        
        return features
    
    def _calculate_skill_similarity(self, jd_skills: List[str], 
                                  resume_skills: List[str]) -> float:
        """Calculate advanced skill similarity score."""
        if not jd_skills:
            return 0.5
        
        # Convert to sets for intersection calculation
        jd_skills_set = set(skill.lower() for skill in jd_skills)
        resume_skills_set = set(skill.lower() for skill in resume_skills)
        
        # Calculate overlap
        intersection = jd_skills_set.intersection(resume_skills_set)
        overlap_ratio = len(intersection) / len(jd_skills_set)
        
        # Bonus for having more skills than required
        extra_skills_bonus = min(0.2, len(resume_skills_set - jd_skills_set) * 0.05)
        
        return min(1.0, overlap_ratio + extra_skills_bonus)
    
    def _calculate_certification_match(self, jd_text: str, resume_text: str) -> float:
        """Calculate certification matching score."""
        jd_lower = jd_text.lower()
        resume_lower = resume_text.lower()
        
        # Common certifications
        certifications = [
            'pmp', 'scrum master', 'aws certified', 'azure certified', 'google cloud',
            'cissp', 'ceh', 'cisa', 'cism', 'itil', 'prince2', 'six sigma',
            'cpa', 'cfa', 'frm', 'series 7', 'series 63', 'caia'
        ]
        
        jd_certs = set(cert for cert in certifications if cert in jd_lower)
        resume_certs = set(cert for cert in certifications if cert in resume_lower)
        
        if not jd_certs:
            return 0.5  # Neutral if no certifications required
        
        overlap = len(jd_certs.intersection(resume_certs))
        return overlap / len(jd_certs)
    
    def _calculate_education_match(self, jd_text: str, resume_text: str) -> float:
        """Calculate education level matching."""
        jd_lower = jd_text.lower()
        resume_lower = resume_text.lower()
        
        education_levels = {
            'phd': 5, 'doctorate': 5, 'doctoral': 5,
            'master': 4, 'mba': 4, 'ms': 4, 'ma': 4,
            'bachelor': 3, 'ba': 3, 'bs': 3, 'btech': 3,
            'associate': 2, 'diploma': 2,
            'certificate': 1
        }
        
        jd_level = max([level for term, level in education_levels.items() 
                       if term in jd_lower], default=3)
        resume_level = max([level for term, level in education_levels.items() 
                           if term in resume_lower], default=3)
        
        # Normalize to 0-1 scale with bonus for exceeding requirements
        if resume_level >= jd_level:
            return min(1.0, resume_level / jd_level * 0.8 + 0.2)
        else:
            return resume_level / jd_level * 0.8
    
    def _calculate_keyword_density(self, jd_text: str, resume_text: str) -> float:
        """Calculate keyword density and relevance."""
        jd_words = set(jd_text.lower().split())
        resume_words = set(resume_text.lower().split())
        
        # Remove common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        jd_words -= stop_words
        resume_words -= stop_words
        
        if not jd_words:
            return 0.5
        
        overlap = len(jd_words.intersection(resume_words))
        return overlap / len(jd_words)
    
    def _calculate_role_alignment(self, jd_text: str, resume_text: str) -> float:
        """Calculate role and responsibility alignment."""
        jd_lower = jd_text.lower()
        resume_lower = resume_text.lower()
        
        role_indicators = {
            'developer': ['develop', 'coding', 'programming', 'software'],
            'analyst': ['analysis', 'requirements', 'business', 'data'],
            'manager': ['manage', 'lead', 'supervise', 'coordinate'],
            'architect': ['design', 'architecture', 'framework', 'system'],
            'engineer': ['engineering', 'technical', 'implementation', 'solution']
        }
        
        jd_roles = []
        for role, indicators in role_indicators.items():
            if role in jd_lower or any(ind in jd_lower for ind in indicators):
                jd_roles.append(role)
        
        resume_roles = []
        for role, indicators in role_indicators.items():
            if role in resume_lower or any(ind in resume_lower for ind in indicators):
                resume_roles.append(role)
        
        if not jd_roles:
            return 0.5
        
        overlap = len(set(jd_roles).intersection(set(resume_roles)))
        return overlap / len(jd_roles)
    
    def _calculate_industry_experience(self, jd_text: str, resume_text: str) -> float:
        """Calculate industry-specific experience alignment."""
        return self.industry_matcher.terminology_mapper.calculate_domain_relevance(
            self.industry_matcher.terminology_mapper.detect_domain(jd_text),
            self.industry_matcher.terminology_mapper.detect_domain(resume_text)
        )
    
    def _detect_leadership_indicators(self, resume_text: str) -> float:
        """Detect leadership experience indicators."""
        resume_lower = resume_text.lower()
        
        leadership_terms = [
            'led team', 'managed team', 'supervised', 'coordinated', 'directed',
            'mentored', 'trained', 'leadership', 'team lead', 'project manager',
            'senior', 'principal', 'architect', 'head of', 'director', 'vp'
        ]
        
        score = sum(1 for term in leadership_terms if term in resume_lower)
        return min(1.0, score / 5)  # Normalize to 0-1
    
    def _assess_technology_freshness(self, resume_text: str) -> float:
        """Assess how current the candidate's technology stack is."""
        resume_lower = resume_text.lower()
        
        modern_tech = [
            'cloud', 'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'microservices',
            'react', 'angular', 'vue', 'node.js', 'python', 'golang', 'rust',
            'machine learning', 'ai', 'blockchain', 'devops', 'ci/cd'
        ]
        
        legacy_penalty = [
            'cobol', 'fortran', 'vb6', 'asp classic', 'flash', 'silverlight'
        ]
        
        modern_score = sum(1 for tech in modern_tech if tech in resume_lower)
        legacy_score = sum(1 for tech in legacy_penalty if tech in resume_lower)
        
        net_score = modern_score - (legacy_score * 0.5)
        return min(1.0, max(0.0, net_score / 10))
    
    def _match_seniority_level(self, jd_text: str, resume_text: str) -> float:
        """Match seniority levels between JD and resume."""
        jd_lower = jd_text.lower()
        resume_lower = resume_text.lower()
        
        seniority_levels = {
            'junior': 1, 'entry': 1, 'associate': 1,
            'mid': 2, 'intermediate': 2, 'regular': 2,
            'senior': 3, 'sr': 3, 'experienced': 3,
            'lead': 4, 'principal': 4, 'staff': 4,
            'architect': 5, 'director': 5, 'vp': 5
        }
        
        jd_level = max([level for term, level in seniority_levels.items() 
                       if term in jd_lower], default=2)
        resume_level = max([level for term, level in seniority_levels.items() 
                           if term in resume_lower], default=2)
        
        # Perfect match gets 1.0, close matches get partial credit
        diff = abs(jd_level - resume_level)
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.8
        elif diff == 2:
            return 0.6
        else:
            return 0.4
    
    def _assess_technical_complexity(self, jd_text: str, resume_text: str) -> float:
        """Assess technical complexity match."""
        jd_lower = jd_text.lower()
        resume_lower = resume_text.lower()
        
        complex_terms = [
            'architecture', 'scalable', 'distributed systems', 'microservices',
            'cloud native', 'kubernetes', 'machine learning', 'ai', 'blockchain',
            'big data', 'real-time', 'high performance', 'enterprise', 'security'
        ]
        
        jd_complexity = sum(1 for term in complex_terms if term in jd_lower)
        resume_complexity = sum(1 for term in complex_terms if term in resume_lower)
        
        if jd_complexity == 0:
            return 0.7  # Neutral for simple roles
        
        # Calculate overlap ratio
        overlap = min(jd_complexity, resume_complexity)
        return min(1.0, overlap / jd_complexity)
    
    def calculate_ensemble_score(self, features: MatchingFeatures) -> Dict[str, float]:
        """Calculate ensemble matching score using multiple algorithms."""
        scores = {}
        
        # Algorithm 1: Skill-focused scoring
        scores['skill_focused'] = (
            features.skill_match * 0.4 +
            features.technical_depth * 0.3 +
            features.certification_match * 0.2 +
            features.technology_freshness * 0.1
        )
        
        # Algorithm 2: Semantic-focused scoring
        scores['semantic_focused'] = (
            features.experience_match * 0.3 +
            features.seniority_match * 0.3 +
            features.role_alignment * 0.2 +
            features.keyword_density * 0.2
        )
        
        # Algorithm 3: Industry-focused scoring
        scores['industry_focused'] = (
            features.domain_relevance * 0.4 +
            features.industry_experience * 0.3 +
            features.certification_match * 0.2 +
            features.education_match * 0.1
        )
        
        # Algorithm 4: Experience-focused scoring
        scores['experience_focused'] = (
            features.experience_match * 0.3 +
            features.leadership_indicators * 0.25 +
            features.seniority_match * 0.25 +
            features.industry_experience * 0.2
        )
        
        # Algorithm 5: Weighted feature combination
        feature_dict = features.to_dict()
        scores['weighted_combination'] = sum(
            feature_dict[feature] * weight 
            for feature, weight in self.feature_weights.items()
            if feature in feature_dict
        )
        
        # Final ensemble score
        ensemble_score = sum(
            scores[algorithm] * weight 
            for algorithm, weight in self.ensemble_weights.items()
            if algorithm in scores
        )
        
        scores['ensemble_final'] = min(1.0, max(0.0, ensemble_score))
        
        return scores
    
    def train_ml_models(self, training_data: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """Train ML models on historical matching data."""
        if len(training_data) < 10:
            logger.warning("Insufficient training data for ML models")
            return {}
        
        # Extract features and targets
        X = []
        y = []
        
        for jd_text, resume_text, target_score in training_data:
            features = self.extract_comprehensive_features(jd_text, resume_text)
            X.append(features.to_array())
            y.append(target_score)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        model_performance = {}
        
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
                predictions = model.predict(X_scaled)
                r2 = r2_score(y, predictions)
                mse = mean_squared_error(y, predictions)
                
                model_performance[name] = {
                    'r2_score': r2,
                    'mse': mse,
                    'trained': True
                }
                
                logger.info(f"Model {name}: R2={r2:.3f}, MSE={mse:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                model_performance[name] = {'trained': False, 'error': str(e)}
        
        self.is_trained = True
        return model_performance
    
    def predict_ml_score(self, features: MatchingFeatures) -> float:
        """Predict matching score using trained ML models."""
        if not self.is_trained:
            return 0.5  # Neutral score if not trained
        
        X = features.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)[0]
                predictions.append(max(0.0, min(1.0, pred)))  # Clamp to [0,1]
                
                # Weight based on model type
                if name == 'random_forest':
                    weights.append(0.4)
                elif name == 'gradient_boost':
                    weights.append(0.4)
                else:
                    weights.append(0.2)
                    
            except Exception as e:
                logger.error(f"Prediction error with {name}: {e}")
        
        if not predictions:
            return 0.5
        
        # Weighted average of predictions
        weighted_sum = sum(p * w for p, w in zip(predictions, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def enhanced_match(self, jd_text: str, resume_text: str) -> Dict[str, Any]:
        """Perform enhanced matching with all optimizations."""
        # Extract comprehensive features
        features = self.extract_comprehensive_features(jd_text, resume_text)
        
        # Calculate ensemble scores
        ensemble_scores = self.calculate_ensemble_score(features)
        
        # Get ML prediction if available
        ml_score = self.predict_ml_score(features) if self.is_trained else None
        
        # Calculate confidence score
        score_variance = np.var(list(ensemble_scores.values()))
        confidence = max(0.0, 1.0 - score_variance * 2)  # Higher variance = lower confidence
        
        return {
            'final_score': ensemble_scores['ensemble_final'],
            'algorithm_scores': ensemble_scores,
            'ml_score': ml_score,
            'features': features.to_dict(),
            'confidence': confidence,
            'feature_breakdown': self._analyze_feature_contributions(features)
        }
    
    def _analyze_feature_contributions(self, features: MatchingFeatures) -> Dict[str, Dict[str, float]]:
        """Analyze which features contribute most to the match."""
        feature_dict = features.to_dict()
        
        # Calculate weighted contributions
        contributions = {}
        total_weighted_score = 0
        
        for feature, value in feature_dict.items():
            if feature in self.feature_weights:
                weight = self.feature_weights[feature]
                contribution = value * weight
                contributions[feature] = {
                    'value': value,
                    'weight': weight,
                    'contribution': contribution,
                    'percentage': 0.0  # Will be calculated below
                }
                total_weighted_score += contribution
        
        # Calculate percentages
        if total_weighted_score > 0:
            for feature in contributions:
                contributions[feature]['percentage'] = (
                    contributions[feature]['contribution'] / total_weighted_score * 100
                )
        
        return contributions


def test_phase4_optimization():
    """Test Phase 4 optimization system."""
    engine = EnsembleMatchingEngine()
    
    # Test case 1: Healthcare IT position
    jd_healthcare = """
    Senior Healthcare IT Business Analyst with 5+ years experience in EHR systems,
    HIPAA compliance, and Epic certification. Master's degree preferred.
    Must have experience leading teams and implementing clinical workflows.
    """
    
    resume_healthcare = """
    Senior Business Analyst with 7 years in healthcare IT. Epic certified with
    extensive EHR implementation experience. MBA from top university. 
    Led teams of 5-8 analysts in HIPAA compliance projects. Strong clinical
    workflow optimization background with modern cloud technologies.
    """
    
    result = engine.enhanced_match(jd_healthcare, resume_healthcare)
    
    print("🧪 Phase 4 Test Results:")
    print(f"Final Score: {result['final_score']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Algorithm Scores: {result['algorithm_scores']}")
    print(f"Top Contributing Features:")
    
    # Show top 5 contributing features
    contributions = result['feature_breakdown']
    sorted_features = sorted(contributions.items(), 
                           key=lambda x: x[1]['contribution'], reverse=True)
    
    for feature, data in sorted_features[:5]:
        print(f"  {feature}: {data['value']:.3f} (weight: {data['weight']:.3f}, "
              f"contribution: {data['percentage']:.1f}%)")


if __name__ == "__main__":
    test_phase4_optimization()