import os
import json
from docx import Document
from config import DATASETS_DIR, DATA_DIR

def extract_text_from_docx(file_path):
    """
    Extract text from a .docx file.
    """
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def process_resume(file_path):
    """
    Process a single resume: extract text and return as dict.
    """
    text = extract_text_from_docx(file_path)
    filename = os.path.basename(file_path)
    return {
        "filename": filename,
        "raw_text": text,
        "file_path": file_path
    }

def process_all_resumes(output_json_path):
    """
    Process all resumes in Datasets/Resumes/ and save to JSON.
    """
    resumes_dir = os.path.join(DATASETS_DIR, 'Resumes')
    resumes = []
    for file in os.listdir(resumes_dir):
        if file.endswith('.docx'):
            file_path = os.path.join(resumes_dir, file)
            resume_data = process_resume(file_path)
            resumes.append(resume_data)
    
    with open(output_json_path, 'w') as f:
        json.dump(resumes, f, indent=4)
    print(f"Processed {len(resumes)} resumes and saved to {output_json_path}")

if __name__ == "__main__":
    output_path = os.path.join(DATA_DIR, 'processed', 'resumes_raw.json')
    process_all_resumes(output_path)
