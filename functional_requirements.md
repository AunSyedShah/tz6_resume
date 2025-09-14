# AI Resume Ranker - Functional Requirements

## 1.2 Proposed Solution

The **AI-powered resume ranking system** automates candidate shortlisting by leveraging **NLP** and intelligent ranking algorithms. The system matches resumes with job descriptions (JDs), reducing manual effort and improving efficiency in candidate selection.

### Workflow
1. **Resume Upload**  
   Applicant uploads a resume (unstructured data).
   
2. **NLP Preprocessing**  
   - Tokenization  
   - Stopword Removal  
   - Stemming  
   - Lemmatization  

   Purpose: Normalize and prepare resume text.

3. **Skill Set Extraction**  
   - Extract technical and domain-specific skills from resumes.  
   - Organize extracted skills into a structured format.  

4. **Job Description Processing (NER)**  
   - Use **Named Entity Recognition (NER)** on JDs.  
   - Identify key expectations: skills, experience, job roles.  

5. **Classification Module**  
   - Classifies resumes based on relevance and alignment with JDs.  

6. **Category-Based Matching**  
   - Compares resumes against job categories.  
   - Ensures most relevant resumes are ranked higher.  

7. **Ranking System**  
   - Integrates classification + matching outputs.  
   - Produces ranked candidate lists for recruiters.  

### Sample Dataset
- **Source:** 228 Word documents (`.docx`), each a resume.  
- **Size:** 24 KB – 90 KB per file.  
- **Content:** Candidate details, skills, education, certifications, work experience.  
- **Profiles:** Full Stack Developers, Business Analysts, Project Managers, Software Engineers.  

Dataset provides the **foundation** for training ML/NLP models.

---

## 1.2.1 Steps to Build the Model

1. **Resume Dataset Ingestion**  
   - Convert resumes to JSON format.  
   - Extract applicant info: education, experience, skills, certifications.  

2. **Text Preprocessing**  
   - Tokenization  
   - Stopword Removal  
   - Stemming & Lemmatization  

3. **Named Entity Recognition (NER)**  
   - Extract entities: skills, roles, education, company names.  

4. **Skillset-Based Extraction**  
   - Isolate resumes matching job-required skills.  

5. **Matching Engine**  
   - Compare resumes & JDs.  
   - Calculate **matching score** using:  
     - Keyword overlap  
     - Semantic similarity  
     - Contextual alignment  

6. **Scoring, Ranking & Shortlisting**  
   - Rank candidates based on scores.  
   - Select top candidates above threshold.  

7. **Output to Recruiters**  
   - Display **ranked, filtered list** of best-fit candidates.  
   - Reduce time-to-hire & support **merit-based selection**.  

---

## 1.3 Purpose of Document

- Provide a **development plan** for the AI Resume Ranker.  
- Ensure **stakeholder alignment** and effective communication.  
- Support **high-quality application delivery**.  
- Help recruiters focus on **high-potential candidates**.  

Audience: **Stakeholders & Developers**.  

---

## 1.4 Scope of Project

The project delivers a **complete ML pipeline** for resume ranking:  
- Automates shortlisting with **NLP techniques**.  
- Extracts **skills, roles, qualifications**.  
- Calculates **relevance scores**.  
- Displays a **ranked candidate list** in a user-friendly interface.  
- Supports recruiter **feedback loops** to improve model accuracy over time.  

---

## 1.6 Functional Requirements

1. **Resume Upload**  
   - Upload single/multiple `.docx` resumes via web interface.  

2. **Job Description Input**  
   - Recruiters can input or upload a JD.  

3. **Candidate Scoring, Ranking & Shortlisting**  
   - Rank resumes using matching engine scores.  
   - Auto-shortlist candidates above threshold.  

4. **Search & Filter**  
   - Search by keywords.  
   - Filter by ranking score, experience, skills, qualifications.  

5. **Resume Parsing**  
   - Extract job-relevant skills & competencies.  
   - Retain only resumes meeting technical requirements.  

6. **Ranked Resume Display**  
   - Display ranked list of top candidates.  
   - Highlight skills, experience, relevance score.  

7. **Resume Download or View**  
   - Recruiters can view/download shortlisted resumes.  

8. **Export Results**  
   - Export candidate lists in **Excel/PDF**.  

9. **User Interface**  
   - Intuitive UI for uploading resumes/JDs.  
   - Display ranked list with highlights.  

10. **Feedback Loop**  
   - Capture recruiter decisions (hired/rejected).  
   - Continuously retrain & improve accuracy.  

---

## 1.8.2 Software Technologies

- **Frontend:** HTML5, Streamlit (or other frontend frameworks)  
- **Backend:** Flask / Django  
- **Data Store:** JSON, TXT, CSV, Word, PDF  
- **Programming/IDE:** Python, R, Jupyter Notebook, Anaconda, Google Colab  
- **Libraries:**  
  - `python-docx`  
  - `PyPDF2`  
  - `NER`  
  - `NLTK`  
  - `regex`  
  - `scikit-learn`  
  - `NumPy`  
  - `Pandas`  
  - `Matplotlib`  

---
