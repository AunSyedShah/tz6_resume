"""
Phase 6: Advanced Search & Filter UI
===================================

This module implements comprehensive search and filtering capabilities
for the resume ranking system with real-time updates and advanced criteria.

Features:
- Real-time keyword search across resume content
- Multi-criteria filtering (score ranges, experience levels, skills)
- Advanced sorting options with multiple sort keys
- Filter persistence and saved search profiles
- Quick filter presets for common scenarios
- Dynamic filter suggestions and auto-complete
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchFilter:
    """Represents a search filter configuration."""
    keyword: str = ""
    score_min: float = 0.0
    score_max: float = 1.0
    experience_min: int = 0
    experience_max: int = 20
    skills_required: List[str] = None
    skills_preferred: List[str] = None
    education_level: str = "any"
    certifications: List[str] = None
    roles: List[str] = None
    domains: List[str] = None
    confidence_min: float = 0.0
    exclude_keywords: List[str] = None
    
    def __post_init__(self):
        if self.skills_required is None:
            self.skills_required = []
        if self.skills_preferred is None:
            self.skills_preferred = []
        if self.certifications is None:
            self.certifications = []
        if self.roles is None:
            self.roles = []
        if self.domains is None:
            self.domains = []
        if self.exclude_keywords is None:
            self.exclude_keywords = []


@dataclass 
class SortCriteria:
    """Represents sorting criteria for results."""
    primary_field: str = "final_score"
    primary_order: str = "desc"  # "asc" or "desc"
    secondary_field: Optional[str] = None
    secondary_order: str = "desc"
    tertiary_field: Optional[str] = None
    tertiary_order: str = "desc"


class AdvancedSearchEngine:
    """Advanced search and filtering engine for candidate results."""
    
    def __init__(self):
        self.search_history = []
        self.saved_filters = {}
        self.quick_presets = self._initialize_quick_presets()
        self.skill_suggestions = set()
        self.role_suggestions = set()
        
    def _initialize_quick_presets(self) -> Dict[str, SearchFilter]:
        """Initialize commonly used filter presets."""
        return {
            "top_candidates": SearchFilter(
                score_min=0.7,
                confidence_min=0.8
            ),
            "senior_roles": SearchFilter(
                experience_min=5,
                roles=["senior", "lead", "principal", "architect"]
            ),
            "entry_level": SearchFilter(
                experience_max=2,
                roles=["junior", "entry", "associate"]
            ),
            "high_confidence": SearchFilter(
                confidence_min=0.9,
                score_min=0.6
            ),
            "technical_roles": SearchFilter(
                skills_required=["programming", "development", "engineering"],
                domains=["technology"]
            ),
            "management_roles": SearchFilter(
                skills_required=["leadership", "management", "team"],
                experience_min=3
            )
        }
    
    def search_candidates(self, candidates: List[Dict], 
                         search_filter: SearchFilter,
                         sort_criteria: SortCriteria = None) -> List[Dict]:
        """Execute advanced search with filtering and sorting."""
        if not candidates:
            return []
        
        # Apply filters
        filtered_candidates = self._apply_filters(candidates, search_filter)
        
        # Apply sorting
        if sort_criteria:
            filtered_candidates = self._apply_sorting(filtered_candidates, sort_criteria)
        
        # Update search history
        self._update_search_history(search_filter, len(filtered_candidates))
        
        return filtered_candidates
    
    def _apply_filters(self, candidates: List[Dict], search_filter: SearchFilter) -> List[Dict]:
        """Apply all search filters to candidate list."""
        filtered = candidates.copy()
        
        # Keyword search
        if search_filter.keyword:
            filtered = self._filter_by_keyword(filtered, search_filter.keyword)
        
        # Exclude keywords
        if search_filter.exclude_keywords:
            for exclude_keyword in search_filter.exclude_keywords:
                filtered = self._filter_exclude_keyword(filtered, exclude_keyword)
        
        # Score range
        filtered = self._filter_by_score_range(
            filtered, search_filter.score_min, search_filter.score_max
        )
        
        # Experience range
        filtered = self._filter_by_experience_range(
            filtered, search_filter.experience_min, search_filter.experience_max
        )
        
        # Required skills
        if search_filter.skills_required:
            filtered = self._filter_by_required_skills(filtered, search_filter.skills_required)
        
        # Preferred skills (boost scoring)
        if search_filter.skills_preferred:
            filtered = self._boost_preferred_skills(filtered, search_filter.skills_preferred)
        
        # Education level
        if search_filter.education_level != "any":
            filtered = self._filter_by_education(filtered, search_filter.education_level)
        
        # Certifications
        if search_filter.certifications:
            filtered = self._filter_by_certifications(filtered, search_filter.certifications)
        
        # Roles
        if search_filter.roles:
            filtered = self._filter_by_roles(filtered, search_filter.roles)
        
        # Domains
        if search_filter.domains:
            filtered = self._filter_by_domains(filtered, search_filter.domains)
        
        # Confidence threshold
        filtered = self._filter_by_confidence(filtered, search_filter.confidence_min)
        
        return filtered
    
    def _filter_by_keyword(self, candidates: List[Dict], keyword: str) -> List[Dict]:
        """Filter candidates by keyword search."""
        if not keyword:
            return candidates
        
        keyword_lower = keyword.lower()
        filtered = []
        
        for candidate in candidates:
            # Search in filename
            filename = candidate.get('filename', '').lower()
            if keyword_lower in filename:
                filtered.append(candidate)
                continue
            
            # Search in skills
            skills = candidate.get('All_Skills', [])
            if any(keyword_lower in skill.lower() for skill in skills):
                filtered.append(candidate)
                continue
            
            # Search in roles
            roles = candidate.get('All_Roles', [])
            if any(keyword_lower in role.lower() for role in roles):
                filtered.append(candidate)
                continue
            
            # Search in raw content if available
            content = candidate.get('content', '') or candidate.get('raw_text', '')
            if keyword_lower in content.lower():
                filtered.append(candidate)
                continue
        
        return filtered
    
    def _filter_exclude_keyword(self, candidates: List[Dict], exclude_keyword: str) -> List[Dict]:
        """Exclude candidates containing specific keywords."""
        if not exclude_keyword:
            return candidates
        
        exclude_lower = exclude_keyword.lower()
        filtered = []
        
        for candidate in candidates:
            # Check filename
            filename = candidate.get('filename', '').lower()
            if exclude_lower in filename:
                continue
            
            # Check content
            content = candidate.get('content', '') or candidate.get('raw_text', '')
            if exclude_lower in content.lower():
                continue
            
            filtered.append(candidate)
        
        return filtered
    
    def _filter_by_score_range(self, candidates: List[Dict], min_score: float, max_score: float) -> List[Dict]:
        """Filter by score range."""
        return [c for c in candidates 
                if min_score <= c.get('final_score', c.get('Score', 0)) <= max_score]
    
    def _filter_by_experience_range(self, candidates: List[Dict], min_exp: int, max_exp: int) -> List[Dict]:
        """Filter by experience range."""
        filtered = []
        for candidate in candidates:
            exp = self._extract_experience_years(candidate)
            if min_exp <= exp <= max_exp:
                filtered.append(candidate)
        return filtered
    
    def _extract_experience_years(self, candidate: Dict) -> int:
        """Extract years of experience from candidate data."""
        # Try to get from features first
        features = candidate.get('features', {})
        if 'experience_years' in features:
            return int(features['experience_years'])
        
        # Try to extract from content
        content = candidate.get('content', '') or candidate.get('raw_text', '')
        if not content:
            return 0
        
        # Look for patterns like "5 years", "5+ years", etc.
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'(\d+)\+?\s*years?\s+experience',
            r'experience\s+of\s+(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?\s+exp'
        ]
        
        content_lower = content.lower()
        for pattern in patterns:
            matches = re.findall(pattern, content_lower)
            if matches:
                return int(matches[0])
        
        return 0
    
    def _filter_by_required_skills(self, candidates: List[Dict], required_skills: List[str]) -> List[Dict]:
        """Filter candidates who have ALL required skills."""
        filtered = []
        required_lower = [skill.lower() for skill in required_skills]
        
        for candidate in candidates:
            candidate_skills = [skill.lower() for skill in candidate.get('All_Skills', [])]
            
            # Check if all required skills are present
            if all(any(req_skill in c_skill for c_skill in candidate_skills) 
                   for req_skill in required_lower):
                filtered.append(candidate)
        
        return filtered
    
    def _boost_preferred_skills(self, candidates: List[Dict], preferred_skills: List[str]) -> List[Dict]:
        """Boost scores for candidates with preferred skills."""
        preferred_lower = [skill.lower() for skill in preferred_skills]
        
        for candidate in candidates:
            candidate_skills = [skill.lower() for skill in candidate.get('All_Skills', [])]
            
            # Count matching preferred skills
            matches = sum(1 for pref_skill in preferred_lower 
                         if any(pref_skill in c_skill for c_skill in candidate_skills))
            
            # Boost score based on preferred skill matches
            if matches > 0:
                boost = min(0.1, matches * 0.02)  # Max 10% boost
                current_score = candidate.get('final_score', candidate.get('Score', 0))
                candidate['boosted_score'] = min(1.0, current_score + boost)
                candidate['preferred_skill_matches'] = matches
        
        return candidates
    
    def _filter_by_education(self, candidates: List[Dict], education_level: str) -> List[Dict]:
        """Filter by education level."""
        education_map = {
            'phd': ['phd', 'doctorate', 'doctoral'],
            'masters': ['master', 'mba', 'ms', 'ma', 'mtech'],
            'bachelors': ['bachelor', 'ba', 'bs', 'btech', 'be'],
            'associates': ['associate', 'diploma']
        }
        
        if education_level not in education_map:
            return candidates
        
        target_terms = education_map[education_level]
        filtered = []
        
        for candidate in candidates:
            content = (candidate.get('content', '') or candidate.get('raw_text', '')).lower()
            if any(term in content for term in target_terms):
                filtered.append(candidate)
        
        return filtered
    
    def _filter_by_certifications(self, candidates: List[Dict], certifications: List[str]) -> List[Dict]:
        """Filter by required certifications."""
        cert_lower = [cert.lower() for cert in certifications]
        filtered = []
        
        for candidate in candidates:
            content = (candidate.get('content', '') or candidate.get('raw_text', '')).lower()
            if any(cert in content for cert in cert_lower):
                filtered.append(candidate)
        
        return filtered
    
    def _filter_by_roles(self, candidates: List[Dict], roles: List[str]) -> List[Dict]:
        """Filter by role types."""
        role_lower = [role.lower() for role in roles]
        filtered = []
        
        for candidate in candidates:
            candidate_roles = [role.lower() for role in candidate.get('All_Roles', [])]
            content = (candidate.get('content', '') or candidate.get('raw_text', '')).lower()
            
            # Check in roles list or content
            if (any(r_role in c_role for r_role in role_lower for c_role in candidate_roles) or
                any(role in content for role in role_lower)):
                filtered.append(candidate)
        
        return filtered
    
    def _filter_by_domains(self, candidates: List[Dict], domains: List[str]) -> List[Dict]:
        """Filter by industry domains."""
        domain_keywords = {
            'healthcare': ['healthcare', 'medical', 'hospital', 'clinical'],
            'finance': ['finance', 'banking', 'financial', 'investment'],
            'technology': ['technology', 'software', 'tech', 'it'],
            'education': ['education', 'academic', 'university', 'school'],
            'manufacturing': ['manufacturing', 'production', 'industrial']
        }
        
        filtered = []
        for candidate in candidates:
            content = (candidate.get('content', '') or candidate.get('raw_text', '')).lower()
            
            for domain in domains:
                domain_lower = domain.lower()
                keywords = domain_keywords.get(domain_lower, [domain_lower])
                
                if any(keyword in content for keyword in keywords):
                    filtered.append(candidate)
                    break
        
        return filtered
    
    def _filter_by_confidence(self, candidates: List[Dict], min_confidence: float) -> List[Dict]:
        """Filter by minimum confidence threshold."""
        return [c for c in candidates 
                if c.get('confidence', 0.8) >= min_confidence]
    
    def _apply_sorting(self, candidates: List[Dict], sort_criteria: SortCriteria) -> List[Dict]:
        """Apply multi-level sorting to candidates."""
        def get_sort_key(candidate: Dict) -> Tuple:
            keys = []
            
            # Primary sort key
            primary_val = self._get_sort_value(candidate, sort_criteria.primary_field)
            keys.append(primary_val if sort_criteria.primary_order == "asc" else -primary_val)
            
            # Secondary sort key
            if sort_criteria.secondary_field:
                secondary_val = self._get_sort_value(candidate, sort_criteria.secondary_field)
                keys.append(secondary_val if sort_criteria.secondary_order == "asc" else -secondary_val)
            
            # Tertiary sort key
            if sort_criteria.tertiary_field:
                tertiary_val = self._get_sort_value(candidate, sort_criteria.tertiary_field)
                keys.append(tertiary_val if sort_criteria.tertiary_order == "asc" else -tertiary_val)
            
            return tuple(keys)
        
        return sorted(candidates, key=get_sort_key)
    
    def _get_sort_value(self, candidate: Dict, field: str) -> float:
        """Get sortable value for a field."""
        if field == "final_score":
            return candidate.get('final_score', candidate.get('Score', 0))
        elif field == "confidence":
            return candidate.get('confidence', 0.8)
        elif field == "experience":
            return self._extract_experience_years(candidate)
        elif field == "skill_match":
            features = candidate.get('features', {})
            return features.get('skill_match', 0)
        elif field == "domain_relevance":
            features = candidate.get('features', {})
            return features.get('domain_relevance', 0)
        elif field == "filename":
            return hash(candidate.get('filename', '')) % 1000  # For alphabetical sort
        else:
            return 0
    
    def _update_search_history(self, search_filter: SearchFilter, result_count: int):
        """Update search history for analytics."""
        history_entry = {
            'timestamp': json.dumps(search_filter.__dict__, default=str),
            'result_count': result_count,
            'filter_summary': self._summarize_filter(search_filter)
        }
        
        self.search_history.append(history_entry)
        
        # Keep only last 100 searches
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]
    
    def _summarize_filter(self, search_filter: SearchFilter) -> str:
        """Create human-readable filter summary."""
        parts = []
        
        if search_filter.keyword:
            parts.append(f"keyword: '{search_filter.keyword}'")
        
        if search_filter.score_min > 0 or search_filter.score_max < 1:
            parts.append(f"score: {search_filter.score_min:.1f}-{search_filter.score_max:.1f}")
        
        if search_filter.experience_min > 0 or search_filter.experience_max < 20:
            parts.append(f"experience: {search_filter.experience_min}-{search_filter.experience_max} years")
        
        if search_filter.skills_required:
            parts.append(f"required skills: {', '.join(search_filter.skills_required[:3])}")
        
        return "; ".join(parts) if parts else "no filters"
    
    def get_search_suggestions(self, candidates: List[Dict], partial_query: str) -> Dict[str, List[str]]:
        """Get search suggestions based on partial query and candidate data."""
        if not partial_query or len(partial_query) < 2:
            return {}
        
        partial_lower = partial_query.lower()
        suggestions = {
            'skills': [],
            'roles': [],
            'keywords': []
        }
        
        # Collect all skills and roles from candidates
        all_skills = set()
        all_roles = set()
        
        for candidate in candidates:
            all_skills.update(candidate.get('All_Skills', []))
            all_roles.update(candidate.get('All_Roles', []))
        
        # Find matching suggestions
        suggestions['skills'] = [skill for skill in all_skills 
                               if partial_lower in skill.lower()][:10]
        suggestions['roles'] = [role for role in all_roles 
                              if partial_lower in role.lower()][:10]
        
        # Common keyword suggestions
        common_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'aws', 'azure',
            'machine learning', 'data science', 'senior', 'lead', 'manager',
            'analyst', 'developer', 'engineer', 'architect'
        ]
        
        suggestions['keywords'] = [kw for kw in common_keywords 
                                 if partial_lower in kw.lower()][:5]
        
        return suggestions
    
    def save_filter_preset(self, name: str, search_filter: SearchFilter):
        """Save a filter configuration as a preset."""
        self.saved_filters[name] = search_filter
    
    def get_filter_preset(self, name: str) -> Optional[SearchFilter]:
        """Get a saved filter preset."""
        return self.saved_filters.get(name) or self.quick_presets.get(name)
    
    def get_available_presets(self) -> Dict[str, str]:
        """Get list of available filter presets with descriptions."""
        presets = {}
        
        # Quick presets
        preset_descriptions = {
            "top_candidates": "High-scoring candidates (70%+ score, 80%+ confidence)",
            "senior_roles": "Senior-level positions (5+ years experience)",
            "entry_level": "Entry-level positions (0-2 years experience)",
            "high_confidence": "High-confidence matches (90%+ confidence)",
            "technical_roles": "Technical/development roles",
            "management_roles": "Management and leadership roles"
        }
        
        presets.update(preset_descriptions)
        
        # Custom saved presets
        for name in self.saved_filters:
            presets[name] = f"Custom filter: {name}"
        
        return presets


def test_advanced_search():
    """Test the advanced search functionality."""
    # Sample candidate data
    sample_candidates = [
        {
            'filename': 'john_senior_java_dev.docx',
            'final_score': 0.85,
            'confidence': 0.92,
            'All_Skills': ['Java', 'Spring Boot', 'AWS', 'Microservices'],
            'All_Roles': ['Senior Developer', 'Tech Lead'],
            'content': 'Senior Java Developer with 8 years of experience in enterprise applications'
        },
        {
            'filename': 'jane_python_analyst.docx',
            'final_score': 0.72,
            'confidence': 0.88,
            'All_Skills': ['Python', 'Data Analysis', 'SQL', 'Machine Learning'],
            'All_Roles': ['Data Analyst', 'Business Analyst'],
            'content': 'Data Analyst with 3 years experience in Python and machine learning'
        },
        {
            'filename': 'bob_entry_developer.docx',
            'final_score': 0.65,
            'confidence': 0.75,
            'All_Skills': ['JavaScript', 'React', 'Node.js'],
            'All_Roles': ['Junior Developer', 'Frontend Developer'],
            'content': 'Recent graduate with 1 year experience in web development'
        }
    ]
    
    search_engine = AdvancedSearchEngine()
    
    # Test 1: Keyword search
    print("Test 1: Keyword search for 'java'")
    filter1 = SearchFilter(keyword="java")
    results1 = search_engine.search_candidates(sample_candidates, filter1)
    print(f"Results: {len(results1)} candidates")
    for r in results1:
        print(f"  - {r['filename']}")
    
    # Test 2: Experience range filter
    print("\nTest 2: Senior roles (5+ years experience)")
    filter2 = search_engine.get_filter_preset("senior_roles")
    results2 = search_engine.search_candidates(sample_candidates, filter2)
    print(f"Results: {len(results2)} candidates")
    
    # Test 3: Score and confidence filter
    print("\nTest 3: Top candidates preset")
    filter3 = search_engine.get_filter_preset("top_candidates")
    results3 = search_engine.search_candidates(sample_candidates, filter3)
    print(f"Results: {len(results3)} candidates")
    
    # Test 4: Multi-criteria search
    print("\nTest 4: Multi-criteria search")
    filter4 = SearchFilter(
        keyword="python",
        score_min=0.7,
        skills_required=["python", "data"]
    )
    results4 = search_engine.search_candidates(sample_candidates, filter4)
    print(f"Results: {len(results4)} candidates")
    
    # Test 5: Search suggestions
    print("\nTest 5: Search suggestions for 'java'")
    suggestions = search_engine.get_search_suggestions(sample_candidates, "java")
    print(f"Suggestions: {suggestions}")


if __name__ == "__main__":
    test_advanced_search()