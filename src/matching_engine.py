import sys
sys.path.append('/workspaces/tz6_resume/src')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import json
from preprocessing import preprocess_text
from nlp_features import extract_skills
from config import DATA_DIR
from semantic_matching import compute_semantic_similarity

def compute_tfidf_similarity(resume_text, jd_text):
    """
    Compute TF-IDF based cosine similarity.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

def skill_overlap_score(resume_skills, jd_skills):
    """
    Compute skill overlap score.
    """
    if not jd_skills:
        return 0
    overlap = set(resume_skills) & set(jd_skills)
    return len(overlap) / len(jd_skills)

def category_matching_score(resume_features, jd_features):
    """
    Simple category-based matching (e.g., role, education).
    """
    score = 0
    if resume_features.get('roles') and jd_features.get('roles'):
        if set(resume_features['roles']) & set(jd_features['roles']):
            score += 0.5
    if resume_features.get('education') and jd_features.get('education'):
        if set(resume_features['education']) & set(jd_features['education']):
            score += 0.3
    return score

def combined_score(resume_data, jd_data):
    """
    Combine similarity scores.
    """
    resume_text = resume_data['processed_text']
    jd_text = jd_data['processed_text']
    tfidf_sim = compute_tfidf_similarity(resume_text, jd_text)
    
    resume_skills = resume_data['features']['skills']
    jd_skills = jd_data['features']['skills']
    skill_score = skill_overlap_score(resume_skills, jd_skills)
    
    cat_score = category_matching_score(resume_data['features'], jd_data['features'])
    
    semantic_sim = compute_semantic_similarity(resume_text, jd_text)
    total_score = 0.4 * semantic_sim + 0.3 * skill_score + 0.3 * cat_score
    return total_score

def rank_resumes(jd_data, resumes_data, top_n=10):
    """
    Rank resumes based on JD.
    """
    scores = []
    for resume in resumes_data:
        score = combined_score(resume, jd_data)
        scores.append((resume['filename'], score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

def train_simple_classifier(resumes_data):
    """
    Train a simple classifier (placeholder, since no labels).
    For demo, use random labels.
    """
    # Placeholder: In real scenario, use labeled data
    X = [resume['processed_text'] for resume in resumes_data[:100]]  # Sample
    y = np.random.randint(0, 2, len(X))  # Random labels
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X_vec = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print(f"Classifier accuracy: {clf.score(X_test, y_test):.2f}")
    return clf, vectorizer

if __name__ == "__main__":
    # Load data
    with open(f"{DATA_DIR}/processed/resumes_processed.json", 'r') as f:
        resumes = json.load(f)
    
    # Sample JD
    sample_jd = "We need a Senior Python Developer with experience in AI, machine learning, and AWS."
    jd_processed = preprocess_text(sample_jd)
    jd_features = {
        'skills': extract_skills(sample_jd),
        'roles': ['Developer'],
        'education': ['Bachelor']
    }
    jd_data = {'processed_text': jd_processed, 'features': jd_features}
    
    # Rank
    top_resumes = rank_resumes(jd_data, resumes[:50])  # Test on subset
    print("Top 5 Resumes:", top_resumes[:5])
    
    # Train classifier
    clf, vec = train_simple_classifier(resumes)
