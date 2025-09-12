import json
import sys
sys.path.append('.')
from src.matching_engine import rank_resumes
from src.job_description import process_job_description

def evaluate_ranking():
    # Load data
    with open("data/processed/resumes_processed.json", 'r') as f:
        resumes = json.load(f)
    
    # Sample JD
    jd_text = "Senior Python Developer with ML experience"
    jd_data = process_job_description(jd_text)
    
    # Rank
    ranked = rank_resumes(jd_data, resumes, top_n=10)
    
    # Simple evaluation: Check if top resumes have relevant skills
    relevant_count = 0
    for fname, score in ranked:
        resume = next(r for r in resumes if r['filename'] == fname)
        skills = resume['features']['skills']
        if 'python' in skills or 'machine learning' in skills:
            relevant_count += 1
    
    precision = relevant_count / len(ranked)
    print(f"Top-10 Precision: {precision:.2f}")
    return precision

if __name__ == "__main__":
    evaluate_ranking()
