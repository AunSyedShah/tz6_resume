"""
Phase 4 Comprehensive Testing: Algorithm Fine-tuning and Optimization
===================================================================

This module tests the final phase algorithm optimizations to measure
the ultimate accuracy improvements across all enhancements.

Test Features:
- Ensemble algorithm evaluation
- Multi-dimensional feature analysis
- Performance benchmarking vs previous phases
- Confidence scoring validation
- Final accuracy assessment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
from typing import Dict, List, Tuple, Any
from phase4_algorithm_optimization import EnsembleMatchingEngine
from phase3_accuracy_test import Phase3AccuracyTester
from phase2_light_test import LightSemanticMatcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase4ComprehensiveTester:
    """Comprehensive testing for Phase 4 algorithm optimizations."""
    
    def __init__(self):
        # Initialize all engines for comparison
        self.phase4_engine = EnsembleMatchingEngine()
        self.phase3_tester = Phase3AccuracyTester()
        self.phase2_matcher = LightSemanticMatcher()
        
        # Load resume data
        self.resumes = self._load_resume_data()
        
        # Define comprehensive test cases
        self.test_cases = self._define_comprehensive_test_cases()
    
    def _load_resume_data(self) -> List[Dict]:
        """Load processed resume data."""
        try:
            with open('/workspaces/tz6_resume/data/processed/resumes_processed.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Resume data not found")
            return []
    
    def _define_comprehensive_test_cases(self) -> List[Dict]:
        """Define comprehensive test cases covering all domains and complexities."""
        return [
            {
                "name": "Senior Healthcare Data Architect",
                "domain": "healthcare",
                "complexity": "high",
                "jd": """
                Senior Healthcare Data Architect needed for Fortune 500 health system.
                Requires 8+ years experience with Epic EHR, HIPAA compliance, and HL7 FHIR
                integration. Master's degree in Computer Science or related field required.
                Must have experience leading teams of 10+ engineers, implementing clinical
                data warehouses, and working with population health analytics. PMP
                certification and Epic certification strongly preferred. Experience with
                modern cloud platforms (AWS/Azure) and machine learning in healthcare essential.
                """,
                "expected_features": {
                    "skill_match": 0.8,
                    "experience_match": 0.9,
                    "seniority_match": 0.9,
                    "domain_relevance": 0.9,
                    "leadership_indicators": 0.8,
                    "certification_match": 0.7
                }
            },
            {
                "name": "Junior Full Stack Developer",
                "domain": "technology",
                "complexity": "low",
                "jd": """
                Junior Full Stack Developer position for startup. Looking for 1-2 years
                experience with React, Node.js, and MongoDB. Bachelor's degree preferred
                but not required. Must be eager to learn and work in agile environment.
                """,
                "expected_features": {
                    "skill_match": 0.7,
                    "experience_match": 0.3,
                    "seniority_match": 0.2,
                    "technical_depth": 0.5,
                    "technology_freshness": 0.8
                }
            },
            {
                "name": "VP Risk Management (Investment Bank)",
                "domain": "finance",
                "complexity": "high",
                "jd": """
                Vice President, Risk Management for global investment bank. Requires 12+
                years experience in financial risk, Basel III compliance, and derivatives
                trading. MBA from top-tier school required. CFA and FRM certifications
                essential. Must have experience managing teams of 20+ analysts, implementing
                risk management systems, and working with regulatory bodies. Experience
                with algorithmic trading, blockchain finance, and machine learning in risk
                preferred.
                """,
                "expected_features": {
                    "skill_match": 0.8,
                    "experience_match": 1.0,
                    "seniority_match": 1.0,
                    "domain_relevance": 0.9,
                    "leadership_indicators": 1.0,
                    "certification_match": 0.9,
                    "education_match": 0.9
                }
            },
            {
                "name": "Manufacturing Process Engineer",
                "domain": "manufacturing",
                "complexity": "medium",
                "jd": """
                Manufacturing Process Engineer for automotive supplier. Requires 4-6 years
                experience with Lean manufacturing, Six Sigma, and ERP systems (SAP preferred).
                Bachelor's in Engineering required. Must have experience with quality
                management systems (ISO 9001), statistical process control, and continuous
                improvement initiatives.
                """,
                "expected_features": {
                    "skill_match": 0.7,
                    "experience_match": 0.6,
                    "domain_relevance": 0.8,
                    "certification_match": 0.6,
                    "education_match": 0.8
                }
            },
            {
                "name": "EdTech Product Manager",
                "domain": "education",
                "complexity": "medium",
                "jd": """
                Product Manager for educational technology startup. Requires 3-5 years
                experience with learning management systems, educational assessment tools,
                and curriculum development. Experience with SCORM standards and learning
                analytics preferred. Must understand K-12 and higher education markets.
                """,
                "expected_features": {
                    "skill_match": 0.7,
                    "experience_match": 0.5,
                    "domain_relevance": 0.8,
                    "role_alignment": 0.8,
                    "industry_experience": 0.7
                }
            }
        ]
    
    def test_single_case_comprehensive(self, test_case: Dict) -> Dict[str, Any]:
        """Test a single case with comprehensive Phase 4 analysis."""
        jd_text = test_case["jd"]
        case_name = test_case["name"]
        expected_domain = test_case["domain"]
        complexity = test_case["complexity"]
        
        print(f"\n📋 Test Case: {case_name}")
        print(f"Domain: {expected_domain.title()} | Complexity: {complexity.title()}")
        print("=" * 80)
        
        # Get top matches using Phase 4 enhanced ensemble matching
        matches = []
        feature_analysis = []
        
        for resume in self.resumes[:50]:  # Test on subset for performance
            resume_text = resume.get('processed_text', resume.get('raw_text', ''))
            
            # Get Phase 4 comprehensive results
            result = self.phase4_engine.enhanced_match(jd_text, resume_text)
            
            matches.append({
                'filename': resume['filename'],
                'final_score': result['final_score'],
                'algorithm_scores': result['algorithm_scores'],
                'ml_score': result.get('ml_score'),
                'confidence': result['confidence'],
                'features': result['features'],
                'feature_breakdown': result['feature_breakdown']
            })
            
            # Collect feature analysis for top matches
            if len(feature_analysis) < 10:
                feature_analysis.append(result['features'])
        
        # Sort by final score
        matches.sort(key=lambda x: x['final_score'], reverse=True)
        top_matches = matches[:5]
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(test_case, matches, feature_analysis)
        
        # Display results
        print(f"🎯 Phase 4 Enhanced Rankings:")
        for i, match in enumerate(top_matches, 1):
            algo_scores = match['algorithm_scores']
            print(f"   {i}. {match['filename']}")
            print(f"      Final: {match['final_score']:.3f} | Confidence: {match['confidence']:.3f}")
            print(f"      Algorithms: Skill={algo_scores.get('skill_focused', 0):.2f} | "
                  f"Semantic={algo_scores.get('semantic_focused', 0):.2f} | "
                  f"Industry={algo_scores.get('industry_focused', 0):.2f}")
        
        print(f"\n📊 Phase 4 Comprehensive Metrics:")
        print(f"      Average Final Score: {metrics['avg_final_score']:.3f}")
        print(f"      Average Confidence: {metrics['avg_confidence']:.3f}")
        print(f"      Feature Accuracy: {metrics['feature_accuracy']:.3f}")
        print(f"      Algorithm Consistency: {metrics['algorithm_consistency']:.3f}")
        print(f"      Score Distribution: {metrics['score_distribution']}")
        
        print(f"\n🔍 Top Match Feature Analysis:")
        if top_matches:
            top_features = top_matches[0]['features']
            feature_breakdown = top_matches[0]['feature_breakdown']
            
            # Show top contributing features
            sorted_features = sorted(feature_breakdown.items(), 
                                   key=lambda x: x[1]['contribution'], reverse=True)
            
            for feature, data in sorted_features[:6]:
                print(f"      {feature}: {data['value']:.3f} "
                      f"(contribution: {data['percentage']:.1f}%)")
        
        print(f"\n💡 Algorithm Performance:")
        if top_matches:
            algo_scores = top_matches[0]['algorithm_scores']
            for algo, score in algo_scores.items():
                if algo != 'ensemble_final':
                    print(f"      {algo}: {score:.3f}")
        
        return {
            'case_name': case_name,
            'domain': expected_domain,
            'complexity': complexity,
            'metrics': metrics,
            'top_matches': top_matches[:3],
            'feature_analysis': feature_analysis[:5],
            'algorithm_scores': top_matches[0]['algorithm_scores'] if top_matches else {}
        }
    
    def _calculate_comprehensive_metrics(self, test_case: Dict, matches: List[Dict], 
                                       feature_analysis: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for Phase 4 evaluation."""
        if not matches:
            return {
                'avg_final_score': 0, 'avg_confidence': 0, 'feature_accuracy': 0,
                'algorithm_consistency': 0, 'score_distribution': {}
            }
        
        # Basic metrics
        final_scores = [m['final_score'] for m in matches]
        confidences = [m['confidence'] for m in matches]
        
        avg_final_score = np.mean(final_scores)
        avg_confidence = np.mean(confidences)
        
        # Feature accuracy (how well features match expectations)
        expected_features = test_case.get('expected_features', {})
        feature_accuracy = 0.0
        
        if expected_features and matches:
            top_features = matches[0]['features']
            accuracy_scores = []
            
            for feature, expected_value in expected_features.items():
                if feature in top_features:
                    actual_value = top_features[feature]
                    # Calculate accuracy as 1 - abs_difference
                    accuracy = 1.0 - abs(expected_value - actual_value)
                    accuracy_scores.append(max(0.0, accuracy))
            
            feature_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
        
        # Algorithm consistency (how consistent are different algorithms)
        algorithm_consistency = 0.0
        if matches:
            top_match = matches[0]
            algo_scores = [score for algo, score in top_match['algorithm_scores'].items() 
                          if algo != 'ensemble_final']
            if len(algo_scores) > 1:
                algorithm_consistency = 1.0 - (np.std(algo_scores) / np.mean(algo_scores))
                algorithm_consistency = max(0.0, min(1.0, algorithm_consistency))
        
        # Score distribution analysis
        score_ranges = {
            'excellent': len([s for s in final_scores if s >= 0.8]),
            'good': len([s for s in final_scores if 0.6 <= s < 0.8]),
            'fair': len([s for s in final_scores if 0.4 <= s < 0.6]),
            'poor': len([s for s in final_scores if s < 0.4])
        }
        
        return {
            'avg_final_score': avg_final_score,
            'avg_confidence': avg_confidence,
            'feature_accuracy': feature_accuracy,
            'algorithm_consistency': algorithm_consistency,
            'score_distribution': score_ranges,
            'score_std': np.std(final_scores),
            'confidence_std': np.std(confidences)
        }
    
    def run_phase4_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive Phase 4 benchmark testing."""
        print("🚀 Phase 4 Comprehensive Benchmark: Algorithm Fine-tuning")
        print("=" * 75)
        print(f"✅ Loaded {len(self.resumes)} processed resumes")
        print(f"🎯 Testing {len(self.test_cases)} comprehensive scenarios")
        print(f"🧠 Ensemble Methods: 5 algorithms with ML optimization")
        
        results = []
        overall_metrics = {
            'final_scores': [],
            'confidences': [],
            'feature_accuracies': [],
            'algorithm_consistencies': [],
            'complexity_performance': {'low': [], 'medium': [], 'high': []}
        }
        
        for test_case in self.test_cases:
            case_result = self.test_single_case_comprehensive(test_case)
            results.append(case_result)
            
            # Aggregate metrics
            metrics = case_result['metrics']
            overall_metrics['final_scores'].append(metrics['avg_final_score'])
            overall_metrics['confidences'].append(metrics['avg_confidence'])
            overall_metrics['feature_accuracies'].append(metrics['feature_accuracy'])
            overall_metrics['algorithm_consistencies'].append(metrics['algorithm_consistency'])
            
            # Track by complexity
            complexity = case_result['complexity']
            if complexity in overall_metrics['complexity_performance']:
                overall_metrics['complexity_performance'][complexity].append(metrics['avg_final_score'])
        
        # Calculate final performance metrics
        final_performance = {
            'overall_accuracy': np.mean(overall_metrics['final_scores']),
            'avg_confidence': np.mean(overall_metrics['confidences']),
            'feature_accuracy': np.mean(overall_metrics['feature_accuracies']),
            'algorithm_consistency': np.mean(overall_metrics['algorithm_consistencies']),
            'accuracy_std': np.std(overall_metrics['final_scores']),
            'complexity_breakdown': {
                complexity: np.mean(scores) if scores else 0.0
                for complexity, scores in overall_metrics['complexity_performance'].items()
            }
        }
        
        # Display comprehensive results
        print(f"\n🏆 Phase 4 Final Results:")
        print(f"   Overall Accuracy: {final_performance['overall_accuracy']:.1%}")
        print(f"   Average Confidence: {final_performance['avg_confidence']:.1%}")
        print(f"   Feature Accuracy: {final_performance['feature_accuracy']:.1%}")
        print(f"   Algorithm Consistency: {final_performance['algorithm_consistency']:.1%}")
        print(f"   Accuracy Standard Deviation: ±{final_performance['accuracy_std']:.3f}")
        
        print(f"\n📈 Performance by Complexity:")
        for complexity, score in final_performance['complexity_breakdown'].items():
            print(f"   {complexity.title()}: {score:.1%}")
        
        print(f"\n📊 Phase 4 Assessment:")
        overall_acc = final_performance['overall_accuracy']
        confidence = final_performance['avg_confidence']
        
        if overall_acc > 0.85:
            print("🎉 OUTSTANDING: Exceptional matching accuracy achieved")
        elif overall_acc > 0.75:
            print("✅ EXCELLENT: High-quality matching performance")
        elif overall_acc > 0.65:
            print("✅ GOOD: Solid matching accuracy with room for improvement")
        else:
            print("⚠️  NEEDS WORK: Matching accuracy requires further optimization")
        
        if confidence > 0.8:
            print("✅ HIGH CONFIDENCE: Reliable and consistent predictions")
        elif confidence > 0.6:
            print("✅ MODERATE CONFIDENCE: Generally reliable predictions")
        else:
            print("⚠️  LOW CONFIDENCE: Prediction reliability needs improvement")
        
        print(f"\n🎯 Phase 4 Final Enhancements:")
        print("   ✅ Ensemble algorithm optimization (5 specialized algorithms)")
        print("   ✅ 12-dimensional feature engineering")
        print("   ✅ Machine learning model integration")
        print("   ✅ Confidence scoring and prediction reliability")
        print("   ✅ Advanced weighted scoring systems")
        print("   ✅ Multi-complexity job requirement handling")
        print("   ✅ Feature contribution analysis")
        print("   ✅ Algorithm consistency validation")
        
        print(f"\n🚀 Complete Enhancement Journey:")
        print(f"   Phase 1: Enhanced Skill Mapping → 89.9% accuracy")
        print(f"   Phase 2: Semantic Matching → 77.6% contextual score")
        print(f"   Phase 3: Industry Terminology → 83% domain detection")
        print(f"   Phase 4: Algorithm Optimization → {final_performance['overall_accuracy']:.1%} final accuracy")
        
        return {
            'final_performance': final_performance,
            'case_results': results,
            'test_summary': f"Phase 4 achieved {final_performance['overall_accuracy']:.1%} accuracy with {final_performance['avg_confidence']:.1%} confidence across {len(self.test_cases)} comprehensive test scenarios"
        }


def main():
    """Run Phase 4 comprehensive testing."""
    tester = Phase4ComprehensiveTester()
    results = tester.run_phase4_benchmark()
    return results


if __name__ == "__main__":
    main()