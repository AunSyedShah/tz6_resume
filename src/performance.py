import time
import sys
sys.path.append('.')
from src.matching_engine import rank_resumes
from src.job_description import process_job_description
import json

def performance_test():
    with open("data/processed/resumes_processed.json", 'r') as f:
        resumes = json.load(f)
    
    jd_text = "Business Analyst with SQL and Agile"
    jd_data = process_job_description(jd_text)
    
    start = time.time()
    ranked = rank_resumes(jd_data, resumes)
    end = time.time()
    
    print(f"Ranking 229 resumes took {end - start:.2f} seconds")
    return end - start

if __name__ == "__main__":
    performance_test()
