"""
Phase 3: Industry-Specific Terminology Enhancement
=================================================

This module implements domain-specific terminology mapping to improve
matching accuracy for specialized industries and domains.

Key Features:
- Healthcare domain terminology and certifications
- Financial services terminology and compliance frameworks
- Technology domain specializations
- Industry-specific frameworks and methodologies
- Certification and accreditation mapping
- Domain context detection and scoring
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class IndustryTermMapping:
    """Represents an industry-specific term mapping."""
    canonical_term: str
    synonyms: List[str]
    domain: str
    weight: float = 1.0
    context_keywords: List[str] = None
    
    def __post_init__(self):
        if self.context_keywords is None:
            self.context_keywords = []


class IndustryTerminologyMapper:
    """Enhanced industry-specific terminology mapping system."""
    
    def __init__(self):
        self.industry_mappings = self._initialize_industry_mappings()
        self.domain_keywords = self._initialize_domain_keywords()
        self.certification_mappings = self._initialize_certification_mappings()
        self.framework_mappings = self._initialize_framework_mappings()
        
    def _initialize_industry_mappings(self) -> Dict[str, List[IndustryTermMapping]]:
        """Initialize comprehensive industry-specific term mappings."""
        mappings = defaultdict(list)
        
        # Healthcare Domain
        healthcare_terms = [
            IndustryTermMapping("electronic health records", ["ehr", "electronic medical records", "emr", "health records", "medical records"], "healthcare", 1.2),
            IndustryTermMapping("health information exchange", ["hie", "health data exchange", "medical data exchange"], "healthcare", 1.1),
            IndustryTermMapping("hipaa compliance", ["hipaa", "health insurance portability", "healthcare privacy", "phi protection"], "healthcare", 1.3),
            IndustryTermMapping("clinical data management", ["cdm", "clinical data", "patient data management", "clinical trials data"], "healthcare", 1.2),
            IndustryTermMapping("medical coding", ["icd-10", "icd-9", "cpt codes", "medical billing codes", "diagnosis codes"], "healthcare", 1.2),
            IndustryTermMapping("healthcare interoperability", ["hl7", "fhir", "healthcare standards", "medical interoperability"], "healthcare", 1.3),
            IndustryTermMapping("telemedicine", ["telehealth", "remote patient monitoring", "virtual care", "digital health"], "healthcare", 1.1),
            IndustryTermMapping("clinical workflows", ["clinical processes", "patient care workflows", "healthcare workflows"], "healthcare", 1.1),
            IndustryTermMapping("healthcare analytics", ["population health", "clinical analytics", "healthcare data science"], "healthcare", 1.2),
            IndustryTermMapping("medical devices", ["healthcare devices", "clinical devices", "biomedical equipment"], "healthcare", 1.1),
        ]
        mappings["healthcare"].extend(healthcare_terms)
        
        # Financial Services Domain
        financial_terms = [
            IndustryTermMapping("sarbanes oxley", ["sox", "sox compliance", "financial reporting compliance"], "finance", 1.3),
            IndustryTermMapping("pci compliance", ["pci dss", "payment card industry", "credit card security"], "finance", 1.2),
            IndustryTermMapping("anti money laundering", ["aml", "money laundering prevention", "financial crime prevention"], "finance", 1.2),
            IndustryTermMapping("know your customer", ["kyc", "customer due diligence", "client verification"], "finance", 1.2),
            IndustryTermMapping("basel compliance", ["basel ii", "basel iii", "banking regulations", "capital requirements"], "finance", 1.2),
            IndustryTermMapping("trading systems", ["algorithmic trading", "high frequency trading", "electronic trading"], "finance", 1.2),
            IndustryTermMapping("risk management", ["credit risk", "market risk", "operational risk", "financial risk"], "finance", 1.1),
            IndustryTermMapping("regulatory reporting", ["financial reporting", "compliance reporting", "regulatory filings"], "finance", 1.2),
            IndustryTermMapping("capital markets", ["investment banking", "securities trading", "equity markets"], "finance", 1.1),
            IndustryTermMapping("derivatives trading", ["options trading", "futures trading", "swaps", "financial derivatives"], "finance", 1.2),
            IndustryTermMapping("fintech", ["financial technology", "digital banking", "mobile payments"], "finance", 1.1),
            IndustryTermMapping("blockchain finance", ["cryptocurrency", "digital assets", "defi", "crypto trading"], "finance", 1.2),
        ]
        mappings["finance"].extend(financial_terms)
        
        # Technology Domains
        tech_terms = [
            IndustryTermMapping("microservices architecture", ["microservices", "service oriented architecture", "distributed systems"], "technology", 1.2),
            IndustryTermMapping("cloud native", ["kubernetes native", "containerized applications", "cloud first"], "technology", 1.2),
            IndustryTermMapping("devops practices", ["ci/cd", "continuous integration", "continuous deployment", "infrastructure as code"], "technology", 1.1),
            IndustryTermMapping("machine learning ops", ["mlops", "ml operations", "model deployment", "ml lifecycle"], "technology", 1.3),
            IndustryTermMapping("data engineering", ["big data", "data pipelines", "etl processes", "data warehousing"], "technology", 1.2),
            IndustryTermMapping("cybersecurity", ["information security", "network security", "application security"], "technology", 1.1),
            IndustryTermMapping("api development", ["rest apis", "graphql", "api design", "web services"], "technology", 1.1),
            IndustryTermMapping("full stack development", ["frontend", "backend", "web development", "application development"], "technology", 1.0),
            IndustryTermMapping("mobile development", ["ios development", "android development", "react native", "flutter"], "technology", 1.1),
            IndustryTermMapping("database administration", ["dba", "database management", "sql optimization", "database design"], "technology", 1.1),
        ]
        mappings["technology"].extend(tech_terms)
        
        # Education Domain
        education_terms = [
            IndustryTermMapping("learning management systems", ["lms", "educational technology", "e-learning platforms"], "education", 1.2),
            IndustryTermMapping("student information systems", ["sis", "student records", "academic management"], "education", 1.2),
            IndustryTermMapping("educational assessment", ["student assessment", "academic evaluation", "learning analytics"], "education", 1.1),
            IndustryTermMapping("curriculum development", ["instructional design", "educational content", "course design"], "education", 1.1),
            IndustryTermMapping("distance learning", ["online learning", "remote education", "virtual classrooms"], "education", 1.1),
        ]
        mappings["education"].extend(education_terms)
        
        # Manufacturing Domain
        manufacturing_terms = [
            IndustryTermMapping("enterprise resource planning", ["erp", "sap", "oracle erp", "manufacturing systems"], "manufacturing", 1.2),
            IndustryTermMapping("supply chain management", ["scm", "logistics", "inventory management", "procurement"], "manufacturing", 1.1),
            IndustryTermMapping("quality management", ["iso 9001", "six sigma", "lean manufacturing", "quality assurance"], "manufacturing", 1.2),
            IndustryTermMapping("manufacturing execution systems", ["mes", "production management", "shop floor systems"], "manufacturing", 1.2),
            IndustryTermMapping("industrial automation", ["plc programming", "scada systems", "process automation"], "manufacturing", 1.2),
        ]
        mappings["manufacturing"].extend(manufacturing_terms)
        
        return mappings
    
    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize domain detection keywords."""
        return {
            "healthcare": [
                "hospital", "clinic", "medical", "healthcare", "patient", "clinical",
                "pharmaceutical", "biotech", "health", "medicine", "doctor", "nurse",
                "therapy", "treatment", "diagnosis", "medical device", "pharma"
            ],
            "finance": [
                "bank", "banking", "financial", "finance", "investment", "trading",
                "insurance", "fintech", "capital", "credit", "loan", "mortgage",
                "securities", "asset management", "wealth", "portfolio", "risk"
            ],
            "technology": [
                "software", "technology", "tech", "development", "programming",
                "engineering", "systems", "platform", "application", "digital",
                "data", "analytics", "cloud", "mobile", "web", "api", "security"
            ],
            "education": [
                "education", "school", "university", "college", "academic",
                "learning", "teaching", "student", "curriculum", "training",
                "educational", "campus", "classroom", "instructor", "e-learning"
            ],
            "manufacturing": [
                "manufacturing", "production", "factory", "plant", "industrial",
                "supply chain", "logistics", "quality", "assembly", "operations",
                "procurement", "inventory", "warehouse", "automation"
            ]
        }
    
    def _initialize_certification_mappings(self) -> Dict[str, List[str]]:
        """Initialize certification and accreditation mappings."""
        return {
            "healthcare": [
                "cphims", "cahims", "rhia", "rhit", "ccs", "cca", "epic certified",
                "cerner certified", "healthcare it certification", "clinical informatics"
            ],
            "finance": [
                "cpa", "cfa", "frm", "pmp", "cisa", "cism", "cissp", "series 7",
                "series 63", "chartered financial analyst", "certified financial planner"
            ],
            "technology": [
                "aws certified", "azure certified", "google cloud certified", "cissp",
                "ceh", "oscp", "ccna", "ccnp", "oracle certified", "microsoft certified",
                "docker certified", "kubernetes certified", "scrum master", "safe"
            ],
            "education": [
                "teaching certification", "instructional design certificate",
                "educational technology certification", "certified trainer"
            ],
            "manufacturing": [
                "six sigma black belt", "lean certification", "pmp", "apics",
                "supply chain certification", "quality management certification"
            ]
        }
    
    def _initialize_framework_mappings(self) -> Dict[str, List[str]]:
        """Initialize framework and methodology mappings."""
        return {
            "healthcare": [
                "hl7", "fhir", "dicom", "snomed", "loinc", "ihe", "meaningful use",
                "hitech", "clinical decision support", "interoperability standards"
            ],
            "finance": [
                "basel", "ifrs", "gaap", "sox", "mifid", "dodd frank", "volcker rule",
                "coso framework", "cobit", "itil", "iso 27001"
            ],
            "technology": [
                "agile", "scrum", "kanban", "devops", "microservices", "restful",
                "soap", "oauth", "saml", "kubernetes", "docker", "terraform"
            ],
            "education": [
                "bloom's taxonomy", "instructional systems design", "addie model",
                "scorm", "xapi", "learning objectives", "competency based learning"
            ],
            "manufacturing": [
                "lean", "six sigma", "kaizen", "5s", "tpm", "oee", "smed",
                "value stream mapping", "statistical process control", "cmmi"
            ]
        }
    
    def detect_domain(self, text: str) -> Dict[str, float]:
        """Detect the primary domain(s) of a job description or resume."""
        text_lower = text.lower()
        domain_scores = defaultdict(float)
        
        # Count domain-specific keywords
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    domain_scores[domain] += 1.0
        
        # Boost scores based on industry-specific terms
        for domain, mappings in self.industry_mappings.items():
            for mapping in mappings:
                # Check canonical term
                if mapping.canonical_term.lower() in text_lower:
                    domain_scores[domain] += mapping.weight
                
                # Check synonyms
                for synonym in mapping.synonyms:
                    if synonym.lower() in text_lower:
                        domain_scores[domain] += mapping.weight * 0.8
        
        # Normalize scores
        total_score = sum(domain_scores.values())
        if total_score > 0:
            domain_scores = {k: v/total_score for k, v in domain_scores.items()}
        
        return dict(domain_scores)
    
    def enhance_skill_extraction(self, text: str, detected_domains: Dict[str, float]) -> Dict[str, float]:
        """Extract and enhance skills with domain-specific context."""
        enhanced_skills = {}
        text_lower = text.lower()
        
        # Get top domains (score > 0.1)
        relevant_domains = [domain for domain, score in detected_domains.items() if score > 0.1]
        
        for domain in relevant_domains:
            domain_weight = detected_domains[domain]
            
            # Extract industry-specific terms
            if domain in self.industry_mappings:
                for mapping in self.industry_mappings[domain]:
                    skill_found = False
                    
                    # Check canonical term
                    if mapping.canonical_term.lower() in text_lower:
                        enhanced_skills[mapping.canonical_term] = mapping.weight * domain_weight
                        skill_found = True
                    
                    # Check synonyms
                    for synonym in mapping.synonyms:
                        if synonym.lower() in text_lower:
                            enhanced_skills[mapping.canonical_term] = max(
                                enhanced_skills.get(mapping.canonical_term, 0),
                                mapping.weight * 0.8 * domain_weight
                            )
                            skill_found = True
                    
                    # Context boost if relevant context keywords are found
                    if skill_found and mapping.context_keywords:
                        context_boost = sum(1 for keyword in mapping.context_keywords 
                                          if keyword.lower() in text_lower) * 0.1
                        enhanced_skills[mapping.canonical_term] *= (1 + context_boost)
            
            # Extract certifications
            if domain in self.certification_mappings:
                for cert in self.certification_mappings[domain]:
                    if cert.lower() in text_lower:
                        enhanced_skills[f"certification: {cert}"] = 1.2 * domain_weight
            
            # Extract frameworks
            if domain in self.framework_mappings:
                for framework in self.framework_mappings[domain]:
                    if framework.lower() in text_lower:
                        enhanced_skills[f"framework: {framework}"] = 1.1 * domain_weight
        
        return enhanced_skills
    
    def calculate_domain_relevance(self, jd_domains: Dict[str, float], 
                                 resume_domains: Dict[str, float]) -> float:
        """Calculate domain relevance score between job description and resume."""
        if not jd_domains or not resume_domains:
            return 0.5  # Neutral score if domains can't be detected
        
        # Calculate overlap score
        overlap_score = 0.0
        for domain, jd_score in jd_domains.items():
            if domain in resume_domains:
                overlap_score += min(jd_score, resume_domains[domain])
        
        # Calculate alignment penalty for mismatched domains
        jd_top_domain = max(jd_domains.items(), key=lambda x: x[1])
        resume_top_domain = max(resume_domains.items(), key=lambda x: x[1])
        
        alignment_bonus = 0.0
        if jd_top_domain[0] == resume_top_domain[0]:
            alignment_bonus = 0.2 * min(jd_top_domain[1], resume_top_domain[1])
        
        return min(1.0, overlap_score + alignment_bonus)


class EnhancedIndustryMatcher:
    """Enhanced matching engine with industry-specific terminology."""
    
    def __init__(self):
        self.terminology_mapper = IndustryTerminologyMapper()
        
    def enhanced_skill_match(self, jd_text: str, resume_text: str) -> Dict[str, float]:
        """Enhanced skill matching with industry context."""
        # Detect domains
        jd_domains = self.terminology_mapper.detect_domain(jd_text)
        resume_domains = self.terminology_mapper.detect_domain(resume_text)
        
        # Extract enhanced skills
        jd_skills = self.terminology_mapper.enhance_skill_extraction(jd_text, jd_domains)
        resume_skills = self.terminology_mapper.enhance_skill_extraction(resume_text, resume_domains)
        
        # Calculate domain relevance
        domain_relevance = self.terminology_mapper.calculate_domain_relevance(jd_domains, resume_domains)
        
        # Calculate skill match score
        if not jd_skills:
            skill_match = 0.5
        else:
            matched_skills = 0.0
            total_weight = sum(jd_skills.values())
            
            for skill, jd_weight in jd_skills.items():
                if skill in resume_skills:
                    skill_strength = min(resume_skills[skill], jd_weight)
                    matched_skills += skill_strength
            
            skill_match = matched_skills / total_weight if total_weight > 0 else 0.5
        
        return {
            'skill_match': skill_match,
            'domain_relevance': domain_relevance,
            'jd_domains': jd_domains,
            'resume_domains': resume_domains,
            'jd_skills': jd_skills,
            'resume_skills': resume_skills,
            'overall_score': (skill_match * 0.7 + domain_relevance * 0.3)
        }


def test_industry_terminology():
    """Test the industry terminology system."""
    matcher = EnhancedIndustryMatcher()
    
    # Test healthcare JD
    healthcare_jd = """
    Healthcare IT Business Analyst position requiring experience with EHR systems,
    HIPAA compliance, and HL7 FHIR standards. Must have clinical workflow experience
    and Epic certification preferred.
    """
    
    healthcare_resume = """
    Senior Business Analyst with 5 years in healthcare IT. Extensive experience with
    Electronic Health Records implementation, HIPAA compliance projects, and Epic
    certified. Strong background in clinical data management and HL7 integration.
    """
    
    result = matcher.enhanced_skill_match(healthcare_jd, healthcare_resume)
    print("Healthcare Test Results:")
    print(f"Skill Match: {result['skill_match']:.3f}")
    print(f"Domain Relevance: {result['domain_relevance']:.3f}")
    print(f"Overall Score: {result['overall_score']:.3f}")
    print(f"JD Domains: {result['jd_domains']}")
    print(f"Resume Domains: {result['resume_domains']}")
    print(f"Matched Skills: {list(result['jd_skills'].keys())}")
    print()
    
    # Test finance JD
    finance_jd = """
    Senior Risk Analyst for investment bank. Requires SOX compliance experience,
    Basel III knowledge, and AML/KYC expertise. CFA certification preferred.
    Trading systems experience a plus.
    """
    
    finance_resume = """
    Risk Management Analyst with CFA certification and 7 years in financial services.
    Expert in Sarbanes-Oxley compliance, anti-money laundering procedures, and
    Basel regulatory frameworks. Experience with algorithmic trading platforms.
    """
    
    result = matcher.enhanced_skill_match(finance_jd, finance_resume)
    print("Finance Test Results:")
    print(f"Skill Match: {result['skill_match']:.3f}")
    print(f"Domain Relevance: {result['domain_relevance']:.3f}")
    print(f"Overall Score: {result['overall_score']:.3f}")
    print(f"JD Domains: {result['jd_domains']}")
    print(f"Resume Domains: {result['resume_domains']}")
    print(f"Matched Skills: {list(result['jd_skills'].keys())}")


if __name__ == "__main__":
    test_industry_terminology()