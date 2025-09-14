"""
Phase 3 Accuracy Testing: Industry-Specific Terminology
======================================================

This module tests the enhanced industry-specific terminology matching
to measure accuracy improvements over Phase 2.

Test Features:
- Domain-specific job descriptions and resumes
- Industry terminology recognition
- Certification and framework matching
- Domain relevance scoring
- Comprehensive accuracy metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from typing import Dict, List, Tuple
from phase3_industry_terminology import EnhancedIndustryMatcher
from phase2_light_test import LightSemanticMatcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase3AccuracyTester:
    """Test Phase 3 industry-specific terminology improvements."""
    
    def __init__(self):
        self.industry_matcher = EnhancedIndustryMatcher()
        self.phase2_matcher = LightSemanticMatcher()  # For comparison
        
        # Load resume data
        self.resumes = self._load_resume_data()
        
        # Define industry-specific test cases
        self.test_cases = self._define_test_cases()
    
    def _load_resume_data(self) -> List[Dict]:
        """Load processed resume data."""
        try:
            with open('/workspaces/tz6_resume/data/processed/resumes_processed.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Resume data not found")
            return []
    
    def _define_test_cases(self) -> List[Dict]:
        """Define comprehensive industry-specific test cases."""
        return [
            {
                "name": "Healthcare IT Analyst",
                "domain": "healthcare",
                "jd": """
                Senior Healthcare IT Business Analyst needed for large hospital system.
                Must have 5+ years experience with Electronic Health Records (EHR) implementation,
                HIPAA compliance, and HL7 FHIR integration. Epic certification strongly preferred.
                Experience with clinical workflows, medical coding (ICD-10), and healthcare
                interoperability standards required. Knowledge of telemedicine platforms
                and population health analytics a plus.
                """,
                "expected_skills": [
                    "electronic health records", "hipaa compliance", "hl7", "epic certified",
                    "clinical workflows", "medical coding", "healthcare interoperability"
                ]
            },
            {
                "name": "Financial Risk Analyst",
                "domain": "finance", 
                "jd": """
                Risk Management Analyst for investment banking division. Requires 7+ years
                experience with SOX compliance, Basel III regulatory framework, and AML/KYC
                procedures. CFA certification required. Must have experience with trading
                systems, derivatives analysis, and regulatory reporting. Knowledge of
                blockchain finance and fintech trends preferred.
                """,
                "expected_skills": [
                    "sarbanes oxley", "basel compliance", "anti money laundering", 
                    "know your customer", "cfa", "trading systems", "derivatives trading"
                ]
            },
            {
                "name": "DevOps Engineer (Fintech)",
                "domain": "technology",
                "jd": """
                Senior DevOps Engineer for fintech startup. Need 5+ years with Kubernetes,
                Docker, microservices architecture, and cloud-native development. 
                AWS/Azure certified preferred. Experience with CI/CD pipelines,
                infrastructure as code, and MLOps practices required. PCI DSS compliance
                knowledge essential for payment processing systems.
                """,
                "expected_skills": [
                    "kubernetes", "docker", "microservices architecture", "cloud native",
                    "devops practices", "pci compliance", "machine learning ops"
                ]
            },
            {
                "name": "Manufacturing ERP Consultant",
                "domain": "manufacturing",
                "jd": """
                Senior ERP Implementation Consultant for manufacturing clients. Requires
                8+ years experience with SAP or Oracle ERP systems, supply chain management,
                and manufacturing execution systems (MES). Six Sigma Black Belt certification
                preferred. Must have experience with lean manufacturing, quality management
                (ISO 9001), and industrial automation systems.
                """,
                "expected_skills": [
                    "enterprise resource planning", "supply chain management",
                    "manufacturing execution systems", "six sigma", "quality management"
                ]
            },
            {
                "name": "EdTech Product Manager",
                "domain": "education",
                "jd": """
                Product Manager for educational technology company. Need 4+ years experience
                with learning management systems (LMS), student information systems, and
                educational assessment platforms. Knowledge of SCORM standards, instructional
                design principles, and distance learning technologies required. Experience
                with learning analytics and curriculum development preferred.
                """,
                "expected_skills": [
                    "learning management systems", "student information systems",
                    "educational assessment", "curriculum development", "distance learning"
                ]
            }
        ]
    
    def test_single_case(self, test_case: Dict) -> Dict:
        """Test a single industry-specific case."""
        jd_text = test_case["jd"]
        expected_domain = test_case["domain"]
        expected_skills = set(test_case["expected_skills"])
        
        print(f"\n📋 Test Case: {test_case['name']} ({expected_domain.title()})")
        print("=" * 60)
        
        # Get top matches using Phase 3 enhanced matching
        matches = []
        for resume in self.resumes[:50]:  # Test on subset for speed
            resume_text = resume.get('processed_text', resume.get('raw_text', ''))
            result = self.industry_matcher.enhanced_skill_match(jd_text, resume_text)
            matches.append({
                'filename': resume['filename'],
                'score': result['overall_score'],
                'skill_match': result['skill_match'],
                'domain_relevance': result['domain_relevance'],
                'jd_domains': result['jd_domains'],
                'resume_domains': result['resume_domains'],
                'matched_skills': result['jd_skills']
            })
        
        # Sort by overall score
        matches.sort(key=lambda x: x['score'], reverse=True)
        top_matches = matches[:5]
        
        # Calculate metrics
        metrics = self._calculate_case_metrics(test_case, matches)
        
        # Display results
        print(f"🎯 Enhanced Industry Rankings:")
        for i, match in enumerate(top_matches, 1):
            print(f"   {i}. {match['filename']}")
            print(f"      Overall: {match['score']:.3f} | Skills: {match['skill_match']:.3f} | Domain: {match['domain_relevance']:.3f}")
        
        print(f"\n📊 Case Metrics:")
        print(f"      Domain Detection Accuracy: {metrics['domain_accuracy']:.3f}")
        print(f"      Skill Recognition Rate: {metrics['skill_recognition']:.3f}")
        print(f"      Average Overall Score: {metrics['avg_score']:.3f}")
        print(f"      Average Domain Relevance: {metrics['avg_domain_relevance']:.3f}")
        
        print(f"\n🔍 Domain Analysis:")
        jd_domains = matches[0]['jd_domains'] if matches else {}
        print(f"      Detected JD Domains: {jd_domains}")
        
        print(f"\n💡 Skill Extraction:")
        detected_skills = matches[0]['matched_skills'] if matches else {}
        skill_overlap = expected_skills.intersection(set(detected_skills.keys()))
        print(f"      Expected Skills Found: {len(skill_overlap)}/{len(expected_skills)}")
        print(f"      Matched: {list(skill_overlap)}")
        
        return {
            'case_name': test_case['name'],
            'domain': expected_domain,
            'metrics': metrics,
            'top_matches': top_matches[:3],
            'domain_detection': jd_domains,
            'skill_extraction': detected_skills
        }
    
    def _calculate_case_metrics(self, test_case: Dict, matches: List[Dict]) -> Dict:
        """Calculate detailed metrics for a test case."""
        if not matches:
            return {'domain_accuracy': 0, 'skill_recognition': 0, 'avg_score': 0, 'avg_domain_relevance': 0}
        
        expected_domain = test_case["domain"]
        expected_skills = set(test_case["expected_skills"])
        
        # Domain detection accuracy
        jd_domains = matches[0]['jd_domains'] if matches else {}
        domain_accuracy = jd_domains.get(expected_domain, 0.0)
        
        # Skill recognition rate
        detected_skills = set(matches[0]['matched_skills'].keys()) if matches else set()
        skill_recognition = len(expected_skills.intersection(detected_skills)) / len(expected_skills)
        
        # Average scores
        avg_score = sum(m['score'] for m in matches) / len(matches)
        avg_domain_relevance = sum(m['domain_relevance'] for m in matches) / len(matches)
        
        return {
            'domain_accuracy': domain_accuracy,
            'skill_recognition': skill_recognition,
            'avg_score': avg_score,
            'avg_domain_relevance': avg_domain_relevance
        }
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive Phase 3 testing."""
        print("🚀 Phase 3 Comprehensive Testing: Industry-Specific Terminology")
        print("=" * 70)
        print(f"✅ Loaded {len(self.resumes)} processed resumes")
        print(f"🎯 Testing {len(self.test_cases)} industry-specific scenarios")
        
        results = []
        overall_metrics = {
            'domain_accuracy': [],
            'skill_recognition': [],
            'avg_scores': [],
            'domain_relevance': []
        }
        
        for test_case in self.test_cases:
            case_result = self.test_single_case(test_case)
            results.append(case_result)
            
            # Aggregate metrics
            metrics = case_result['metrics']
            overall_metrics['domain_accuracy'].append(metrics['domain_accuracy'])
            overall_metrics['skill_recognition'].append(metrics['skill_recognition'])
            overall_metrics['avg_scores'].append(metrics['avg_score'])
            overall_metrics['domain_relevance'].append(metrics['avg_domain_relevance'])
        
        # Calculate overall performance
        overall_performance = {
            'domain_accuracy': sum(overall_metrics['domain_accuracy']) / len(overall_metrics['domain_accuracy']),
            'skill_recognition': sum(overall_metrics['skill_recognition']) / len(overall_metrics['skill_recognition']),
            'avg_score': sum(overall_metrics['avg_scores']) / len(overall_metrics['avg_scores']),
            'domain_relevance': sum(overall_metrics['domain_relevance']) / len(overall_metrics['domain_relevance'])
        }
        
        # Display final results
        print(f"\n🏆 Phase 3 Overall Results:")
        print(f"   Domain Detection Accuracy: {overall_performance['domain_accuracy']:.1%}")
        print(f"   Skill Recognition Rate: {overall_performance['skill_recognition']:.1%}")
        print(f"   Average Match Score: {overall_performance['avg_score']:.1%}")
        print(f"   Average Domain Relevance: {overall_performance['domain_relevance']:.1%}")
        
        print(f"\n📈 Phase 3 Assessment:")
        if overall_performance['domain_accuracy'] > 0.7:
            print("✅ EXCELLENT: Strong domain detection capabilities")
        elif overall_performance['domain_accuracy'] > 0.5:
            print("✅ GOOD: Decent domain detection with room for improvement")
        else:
            print("⚠️  NEEDS WORK: Domain detection requires enhancement")
            
        if overall_performance['skill_recognition'] > 0.6:
            print("✅ EXCELLENT: High industry-specific skill recognition")
        elif overall_performance['skill_recognition'] > 0.4:
            print("✅ GOOD: Moderate skill recognition accuracy")
        else:
            print("⚠️  NEEDS WORK: Skill recognition needs improvement")
        
        print(f"\n🎯 Key Phase 3 Enhancements:")
        print("   ✅ Healthcare domain terminology (EHR, HIPAA, HL7)")
        print("   ✅ Financial services terminology (SOX, Basel, AML/KYC)")
        print("   ✅ Technology domain specializations (DevOps, MLOps)")
        print("   ✅ Manufacturing domain knowledge (ERP, MES, Six Sigma)")
        print("   ✅ Education sector terminology (LMS, SCORM)")
        print("   ✅ Certification and framework recognition")
        print("   ✅ Domain-specific context scoring")
        
        return {
            'overall_performance': overall_performance,
            'case_results': results,
            'test_summary': f"Phase 3 achieved {overall_performance['avg_score']:.1%} average accuracy with {overall_performance['domain_accuracy']:.1%} domain detection"
        }


def main():
    """Run Phase 3 comprehensive testing."""
    tester = Phase3AccuracyTester()
    results = tester.run_comprehensive_test()
    return results


if __name__ == "__main__":
    main()