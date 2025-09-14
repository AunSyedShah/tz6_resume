"""
Simple UI Test Script
====================

Test the simplified UI with just two core functionalities:
1. Upload job description (text or file)
2. Upload multiple resumes and rank them by relevance
"""

import requests
import os
import tempfile
from pathlib import Path

def create_sample_files():
    """Create sample job description and resume files for testing."""
    # Sample job description
    jd_content = """
    Job Title: Senior Python Developer
    
    We are looking for a Senior Python Developer with 5+ years of experience.
    
    Required Skills:
    - Python programming
    - Django or Flask framework
    - Database design (PostgreSQL, MySQL)
    - REST API development
    - Git version control
    - Agile development methodology
    
    Preferred Skills:
    - Machine Learning experience
    - AWS cloud services
    - Docker containerization
    - React/JavaScript frontend experience
    
    Responsibilities:
    - Develop and maintain web applications
    - Design database schemas
    - Write clean, maintainable code
    - Collaborate with cross-functional teams
    - Mentor junior developers
    """
    
    # Sample resumes
    resumes = {
        'john_python_expert.txt': """
        John Smith
        Senior Python Developer
        
        Experience: 7 years
        
        Skills:
        - Python (Expert level)
        - Django, Flask
        - PostgreSQL, MySQL
        - REST API development
        - AWS, Docker
        - Machine Learning (TensorFlow, scikit-learn)
        - Git, Agile methodologies
        
        Experience:
        - Lead Python Developer at TechCorp (3 years)
        - Senior Python Developer at StartupXYZ (4 years)
        
        Education:
        - Master's in Computer Science
        """,
        
        'sarah_fullstack.txt': """
        Sarah Johnson
        Full Stack Developer
        
        Experience: 4 years
        
        Skills:
        - Python, JavaScript
        - Flask, React
        - MySQL, MongoDB
        - REST APIs
        - Git
        - Docker basics
        
        Experience:
        - Full Stack Developer at WebSolutions (4 years)
        
        Education:
        - Bachelor's in Software Engineering
        """,
        
        'mike_java_dev.txt': """
        Mike Wilson
        Java Developer
        
        Experience: 6 years
        
        Skills:
        - Java (Primary)
        - Spring Boot
        - MySQL, Oracle
        - REST APIs
        - Maven, Git
        - Some Python knowledge
        
        Experience:
        - Senior Java Developer at Enterprise Corp (6 years)
        
        Education:
        - Bachelor's in Computer Science
        """,
        
        'alex_entry_level.txt': """
        Alex Brown
        Junior Developer
        
        Experience: 1 year
        
        Skills:
        - Python (Learning)
        - Basic Flask
        - HTML, CSS, JavaScript
        - Git basics
        
        Experience:
        - Junior Developer at Local Agency (1 year)
        - Internship at TechStart (6 months)
        
        Education:
        - Bachelor's in Information Technology
        - Recent graduate
        """
    }
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Write job description file
    jd_file = temp_dir / "job_description.txt"
    with open(jd_file, 'w', encoding='utf-8') as f:
        f.write(jd_content)
    
    # Write resume files
    resume_files = []
    for filename, content in resumes.items():
        resume_file = temp_dir / filename
        with open(resume_file, 'w', encoding='utf-8') as f:
            f.write(content)
        resume_files.append(resume_file)
    
    return jd_file, resume_files, temp_dir


def test_flask_app_locally():
    """Test the Flask app by running it and checking the response."""
    print("🧪 Testing Simplified UI Flask Application")
    print("=" * 50)
    
    # Test 1: Check if the main page loads
    print("1️⃣ Testing main page load...")
    try:
        import sys
        sys.path.append('/workspaces/tz6_resume/src')
        
        from flask_app import app
        
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("   ✅ Main page loads successfully")
                
                # Check if key elements are present
                html_content = response.data.decode('utf-8')
                if 'AI Resume Ranker' in html_content:
                    print("   ✅ Title is present")
                if 'Step 1: Upload Job Description' in html_content:
                    print("   ✅ Job description upload section present")
                if 'Step 2: Upload Resumes' in html_content:
                    print("   ✅ Resume upload section present")
                    
            else:
                print(f"   ❌ Main page failed to load: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"   ❌ Error testing main page: {str(e)}")
        return False
    
    # Test 2: Test job description upload (text)
    print("\n2️⃣ Testing job description text upload...")
    try:
        with app.test_client() as client:
            response = client.post('/upload_jd', data={
                'jd_text': 'Sample job description for Python developer with 5+ years experience.'
            }, follow_redirects=True)
            
            if response.status_code == 200:
                print("   ✅ Job description text upload works")
            else:
                print("   ❌ Job description text upload failed")
                
    except Exception as e:
        print(f"   ❌ Error testing JD text upload: {str(e)}")
    
    # Test 3: Test file processing functions
    print("\n3️⃣ Testing file processing functions...")
    try:
        from data_ingestion import extract_text_from_file
        
        # Create sample files
        jd_file, resume_files, temp_dir = create_sample_files()
        
        # Test text extraction
        jd_text = extract_text_from_file(str(jd_file))
        if jd_text and len(jd_text) > 50:
            print("   ✅ Job description text extraction works")
        
        # Test resume processing
        from data_ingestion import process_resume
        
        processed_resumes = []
        for resume_file in resume_files:
            processed = process_resume(str(resume_file))
            if processed and processed.get('raw_text'):
                processed_resumes.append(processed)
        
        if len(processed_resumes) == len(resume_files):
            print(f"   ✅ All {len(resume_files)} resumes processed successfully")
        
        # Test ranking
        from matching_engine import combined_score
        from nlp_features import advanced_nlp_features
        
        scores = []
        for resume in processed_resumes:
            try:
                # Extract features
                features = advanced_nlp_features(resume['raw_text'])
                resume.update(features)
                
                # Calculate score
                score = combined_score(jd_text, resume)
                resume['Score'] = score
                scores.append((resume['filename'], score))
            except Exception as e:
                print(f"   ⚠️ Error processing {resume['filename']}: {str(e)}")
        
        if scores:
            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)
            print("   ✅ Resume ranking successful:")
            for i, (filename, score) in enumerate(scores, 1):
                print(f"      {i}. {filename}: {score:.3f}")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"   ❌ Error testing file processing: {str(e)}")
    
    print("\n✅ Simplified UI testing complete!")
    return True


if __name__ == "__main__":
    test_flask_app_locally()