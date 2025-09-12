import json
from matching_engine import rank_resumes
from job_description import process_job_description
from config import DATA_DIR

def main():
    # Load resumes
    with open(f"{DATA_DIR}/processed/resumes_processed.json", 'r') as f:
        resumes = json.load(f)
    
    # Process JD
    jd_text = "Looking for a Business Analyst with SQL, Agile, and project management skills. Bachelor's degree preferred."
    jd_data = process_job_description(jd_text)
    
    # Rank all resumes
    ranked = rank_resumes(jd_data, resumes)
    
    # Save results
    results = [{"filename": fname, "score": float(score)} for fname, score in ranked]
    with open(f"{DATA_DIR}/processed/ranked_resumes.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Ranked {len(results)} resumes. Top 5:")
    for i, res in enumerate(results[:5]):
        print(f"{i+1}. {res['filename']}: {res['score']:.2f}")

if __name__ == "__main__":
    main()
