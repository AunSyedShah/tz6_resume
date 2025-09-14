# AI Resume Ranker - README

## Overview
This AI-powered resume ranking system evaluates resumes against job descriptions using NLP and ML techniques. It follows a 7-step pipeline for ingestion, processing, matching, and output.

## Features
- **Resume Upload**: Multiple .docx files.
- **JD Input**: Text or file upload.
- **Scoring & Ranking**: Semantic similarity, skills overlap, experience.
- **Shortlisting**: Threshold-based filtering.
- **Search & Filters**: Keywords, skills, roles, experience range.
- **Parsing**: Extracts skills, roles, education, experience.
- **Display**: Highlighted results with expanders.
- **Downloads**: CSV, Excel, PDF, and original .docx.
- **Feedback**: Ratings for model improvement.

## Installation
1. Clone the repo.
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python -m spacy download en_core_web_sm`
5. `python src/process_pipeline.py` (to process base data)
6. `streamlit run src/app.py`

## Usage
- Upload resumes and JD.
- Evaluate and filter results.
- Download reports.

## Alignment
- Fully compliant with provided steps and functional requirements.
