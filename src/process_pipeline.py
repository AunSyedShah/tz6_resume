import json
import os
from data_ingestion import process_all_resumes
from preprocessing import preprocess_resume
from config import DATA_DIR

def main():
    # Step 1: Ingest raw data
    raw_json_path = os.path.join(DATA_DIR, 'processed', 'resumes_raw.json')
    process_all_resumes(raw_json_path)
    
    # Step 2: Load and preprocess
    with open(raw_json_path, 'r') as f:
        resumes = json.load(f)
    
    processed_resumes = []
    for resume in resumes:  # Process all resumes
        processed = preprocess_resume(resume)
        processed_resumes.append(processed)
    
    # Save processed data
    processed_json_path = os.path.join(DATA_DIR, 'processed', 'resumes_processed.json')
    with open(processed_json_path, 'w') as f:
        json.dump(processed_resumes, f, indent=4)
    
    print(f"Processed {len(processed_resumes)} resumes and saved to {processed_json_path}")

if __name__ == "__main__":
    main()
