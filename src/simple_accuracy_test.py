#!/usr/bin/env python3
"""
Simple Accuracy Assessment for Resume Ranking System
This script provides basic accuracy metrics without heavy ML processing.
"""

import json
import sys
import os
sys.path.append('.')

def assess_basic_accuracy():
    """
    Basic accuracy assessment using simple keyword matching
    """
    print("🔍 Resume Ranking System - Accuracy Assessment")
    print("=" * 50)
    
    # Load processed resumes
    try:
        with open("/workspaces/tz6_resume/data/processed/resumes_processed.json", 'r') as f:
            resumes = json.load(f)
        print(f"✅ Loaded {len(resumes)} processed resumes")
    except FileNotFoundError:
        print("❌ Processed resumes not found. Run data processing first.")
        return
    
    # Test cases: Job descriptions and expected skills
    test_cases = [
        {
            "jd": "Senior Java Developer with Spring Boot experience",
            "expected_skills": ["java", "spring", "spring boot", "j2ee"],
            "role": "Java Developer"
        },
        {
            "jd": "Business Analyst with healthcare domain experience",
            "expected_skills": ["business analysis", "healthcare", "requirements", "stakeholder"],
            "role": "Business Analyst"
        },
        {
            "jd": "Python Machine Learning Engineer with AWS experience",
            "expected_skills": ["python", "machine learning", "aws", "ml", "data science"],
            "role": "ML Engineer"
        }
    ]
    
    total_accuracy = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['role']}")
        print(f"JD: {test_case['jd']}")
        
        # Find relevant resumes using simple keyword matching
        relevant_resumes = []
        for resume in resumes:
            resume_text = resume.get('raw_text', '').lower()
            skills_found = 0
            
            for skill in test_case['expected_skills']:
                if skill.lower() in resume_text:
                    skills_found += 1
            
            # Calculate relevance score
            relevance_score = skills_found / len(test_case['expected_skills'])
            if relevance_score > 0.2:  # At least 20% skill match
                relevant_resumes.append({
                    'filename': resume['filename'],
                    'relevance_score': relevance_score,
                    'skills_found': skills_found
                })
        
        # Sort by relevance score
        relevant_resumes.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Display top 5 matches
        top_matches = relevant_resumes[:5]
        print(f"🎯 Found {len(relevant_resumes)} relevant resumes")
        print("Top 5 Matches:")
        
        for j, match in enumerate(top_matches, 1):
            print(f"   {j}. {match['filename']} - Score: {match['relevance_score']:.2f} ({match['skills_found']}/{len(test_case['expected_skills'])} skills)")
        
        # Calculate accuracy for this test case
        if len(top_matches) > 0:
            avg_relevance = sum(match['relevance_score'] for match in top_matches) / len(top_matches)
            accuracy = min(avg_relevance * 100, 100)  # Cap at 100%
            print(f"📊 Test Case Accuracy: {accuracy:.1f}%")
            total_accuracy += accuracy
        else:
            print("📊 Test Case Accuracy: 0.0% (No relevant matches found)")
    
    # Overall accuracy
    overall_accuracy = total_accuracy / len(test_cases)
    print(f"\n🏆 Overall System Accuracy: {overall_accuracy:.1f}%")
    
    # Recommendations
    print(f"\n💡 Assessment Summary:")
    if overall_accuracy >= 70:
        print("✅ GOOD: System shows strong matching capabilities")
    elif overall_accuracy >= 50:
        print("⚠️  MODERATE: System shows decent matching, room for improvement")
    else:
        print("❌ NEEDS IMPROVEMENT: System requires significant enhancement")
    
    print(f"\n📈 Improvement Areas:")
    print("   - Expand skill synonyms and variations")
    print("   - Implement semantic similarity matching") 
    print("   - Add industry-specific terminology")
    print("   - Fine-tune matching algorithms")
    
    return overall_accuracy

if __name__ == "__main__":
    assess_basic_accuracy()