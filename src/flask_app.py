# AI Resume Ranker - Flask Web Application
from flask import Flask, render_template, request, session, jsonify, send_file, flash, redirect, url_for
import pandas as pd
import sys
import os
import tempfile
import json
from pathlib import Path
from werkzeug.utils import secure_filename
from io import BytesIO, StringIO

# Add src to path for imports
sys.path.append('/workspaces/tz6_resume/src')

from job_description import process_job_description
from data_ingestion import process_resume, extract_text_from_docx
from preprocessing import preprocess_resume
from matching_engine import combined_score
from advanced_semantic_matching import AdvancedSemanticMatcher
from feedback import save_feedback
from config_flask import config

app = Flask(__name__)

# Load configuration
config_name = os.environ.get('FLASK_CONFIG', 'development')
app.config.from_object(config[config_name])

# Configuration
UPLOAD_FOLDER = Path('/tmp/resume_uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Template filters
@app.template_filter('get_filtered_results')
def get_filtered_results_filter(dummy):
    """Template filter to get filtered results"""
    return get_filtered_results()

@app.context_processor
def utility_processor():
    """Make utility functions available in templates"""
    return dict(get_filtered_results=get_filtered_results)

@app.route('/')
def index():
    """Main page displaying the resume ranking interface"""
    # Initialize session variables if not present
    if 'jd_text' not in session:
        session['jd_text'] = ""
    if 'jd_option' not in session:
        session['jd_option'] = "text"
    if 'uploaded_files' not in session:
        session['uploaded_files'] = []
    if 'results' not in session:
        session['results'] = []
    if 'evaluation_done' not in session:
        session['evaluation_done'] = False
    if 'use_advanced_matching' not in session:
        session['use_advanced_matching'] = False
    if 'embedding_model' not in session:
        session['embedding_model'] = "all-mpnet-base-v2"
    if 'threshold' not in session:
        session['threshold'] = 0.5
    if 'filter_exp_min' not in session:
        session['filter_exp_min'] = 0
    if 'filter_exp_max' not in session:
        session['filter_exp_max'] = 10
    if 'search_keyword' not in session:
        session['search_keyword'] = ""
    if 'filter_skill' not in session:
        session['filter_skill'] = "All"
    if 'filter_role' not in session:
        session['filter_role'] = "All"
    
    return render_template('index.html')

@app.route('/upload_jd', methods=['POST'])
def upload_jd():
    """Handle job description upload and processing"""
    try:
        jd_option = request.form.get('jd_option', 'text')
        session['jd_option'] = jd_option
        
        if jd_option == 'text':
            jd_text = request.form.get('jd_text', '').strip()
        else:  # file upload
            if 'jd_file' not in request.files:
                flash('No file selected', 'error')
                return redirect(url_for('index'))
            
            file = request.files['jd_file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(url_for('index'))
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                temp_path = os.path.join(tempfile.gettempdir(), filename)
                file.save(temp_path)
                
                if filename.endswith('.txt'):
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        jd_text = f.read()
                elif filename.endswith('.docx'):
                    jd_text = extract_text_from_docx(temp_path)
                
                os.remove(temp_path)  # Clean up temp file
            else:
                flash('Invalid file type. Please upload .txt or .docx files.', 'error')
                return redirect(url_for('index'))
        
        session['jd_text'] = jd_text
        flash('Job description loaded successfully!', 'success')
        
    except Exception as e:
        flash(f'Error processing job description: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/upload_resumes', methods=['POST'])
def upload_resumes():
    """Handle resume file uploads"""
    try:
        if 'resume_files' not in request.files:
            flash('No files selected', 'error')
            return redirect(url_for('index'))
        
        files = request.files.getlist('resume_files')
        uploaded_files = []
        
        for file in files:
            if file.filename == '':
                continue
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                uploaded_files.append({
                    'name': filename,
                    'path': file_path
                })
            else:
                flash(f'Invalid file type for {file.filename}. Please upload .docx files.', 'warning')
        
        session['uploaded_files'] = uploaded_files
        flash(f'{len(uploaded_files)} resume(s) uploaded successfully!', 'success')
        
    except Exception as e:
        flash(f'Error uploading resumes: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/update_config', methods=['POST'])
def update_config():
    """Update matching configuration"""
    try:
        use_advanced = request.form.get('use_advanced_matching') == 'on'
        embedding_model = request.form.get('embedding_model', 'all-mpnet-base-v2')
        
        session['use_advanced_matching'] = use_advanced
        session['embedding_model'] = embedding_model
        
        flash('Configuration updated successfully!', 'success')
        
    except Exception as e:
        flash(f'Error updating configuration: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/evaluate', methods=['POST'])
def evaluate_resumes():
    """Evaluate resumes against job description"""
    try:
        if not session.get('jd_text', '').strip():
            flash('Please enter a job description first.', 'error')
            return redirect(url_for('index'))
        
        if not session.get('uploaded_files'):
            flash('Please upload at least one resume.', 'error')
            return redirect(url_for('index'))
        
        # Process JD
        jd_data = process_job_description(session['jd_text'])
        session['jd_data'] = jd_data
        
        # Process all resumes
        processed_resumes = []
        for file_info in session['uploaded_files']:
            try:
                resume_data = process_resume(file_info['path'])
                resume_data = preprocess_resume(resume_data)
                resume_data['filename'] = file_info['name']
                processed_resumes.append(resume_data)
            except Exception as e:
                flash(f'Error processing {file_info["name"]}: {str(e)}', 'warning')
                continue
        
        if not processed_resumes:
            flash('No resumes could be processed successfully.', 'error')
            return redirect(url_for('index'))
        
        # Evaluate based on selected method
        results = []
        
        if session.get('use_advanced_matching', False):
            # Use advanced semantic matching
            try:
                # Initialize advanced matcher
                matcher = AdvancedSemanticMatcher(model_name=session['embedding_model'])
                
                # Prepare resume data for FAISS
                resume_data_for_faiss = []
                for i, resume in enumerate(processed_resumes):
                    text_content = f"{resume['raw_text']} {' '.join(resume['features'].get('skills', []))}"
                    resume_data_for_faiss.append({
                        'text': text_content,
                        'filename': resume['filename'],
                        'index': i
                    })
                
                # Build FAISS index
                matcher.build_faiss_index(resume_data_for_faiss)
                
                # Score resumes
                jd_combined = f"{jd_data['raw_text']} {' '.join(jd_data['features']['skills'])}"
                scoring_results = matcher.batch_score_resumes(processed_resumes, jd_combined)
                
                # Create results with advanced scores
                for i, resume in enumerate(processed_resumes):
                    if i < len(scoring_results):
                        score_result = scoring_results[i]
                        semantic_score = score_result['advanced_scores']['advanced_score'] * 100
                    else:
                        semantic_score = 50.0
                    
                    results.append({
                        "Filename": resume['filename'],
                        "Score": round(semantic_score, 2),
                        "Semantic Score": round(semantic_score, 2),
                        "Method": "Advanced Semantic (FAISS)",
                        "Skills": ", ".join(resume['features']['skills'][:5]) if resume['features']['skills'] else "No skills detected",
                        "All_Skills": resume['features']['skills'] if resume['features']['skills'] else [],
                        "Roles": ", ".join(resume['features']['roles']) if resume['features']['roles'] else "No roles detected",
                        "All_Roles": resume['features']['roles'] if resume['features']['roles'] else [],
                        "Education": ", ".join(resume['features']['education']) if resume['features']['education'] else "No education detected",
                        "Experience": resume['features'].get('experience_years', 0)
                    })
                
                flash(f'Advanced semantic ranking completed using {session["embedding_model"]}!', 'success')
                
            except Exception as e:
                flash(f'Error in advanced matching: {str(e)}. Falling back to traditional matching.', 'warning')
                # Fall back to traditional matching
                session['use_advanced_matching'] = False
        
        if not session.get('use_advanced_matching', False):
            # Use traditional scoring
            for resume_data in processed_resumes:
                score = combined_score(resume_data, jd_data)
                results.append({
                    "Filename": resume_data['filename'],
                    "Score": round(float(score), 2),
                    "Method": "Traditional (TF-IDF + Rules)",
                    "Skills": ", ".join(resume_data['features']['skills'][:5]) if resume_data['features']['skills'] else "No skills detected",
                    "All_Skills": resume_data['features']['skills'] if resume_data['features']['skills'] else [],
                    "Roles": ", ".join(resume_data['features']['roles']) if resume_data['features']['roles'] else "No roles detected",
                    "All_Roles": resume_data['features']['roles'] if resume_data['features']['roles'] else [],
                    "Education": ", ".join(resume_data['features']['education']) if resume_data['features']['education'] else "No education detected",
                    "Experience": resume_data['features'].get('experience_years', 0)
                })
            
            flash('Traditional scoring completed!', 'success')
        
        # Sort results by score
        results.sort(key=lambda x: x['Score'], reverse=True)
        
        session['results'] = results
        session['evaluation_done'] = True
        
    except Exception as e:
        flash(f'Error during evaluation: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/filter_results', methods=['POST'])
def filter_results():
    """Filter and update results based on criteria"""
    try:
        # Get filter parameters
        threshold = float(request.form.get('threshold', 0.5))
        exp_min = int(request.form.get('filter_exp_min', 0))
        exp_max = int(request.form.get('filter_exp_max', 10))
        search_keyword = request.form.get('search_keyword', '').strip()
        filter_skill = request.form.get('filter_skill', 'All')
        filter_role = request.form.get('filter_role', 'All')
        
        # Update session
        session['threshold'] = threshold
        session['filter_exp_min'] = exp_min
        session['filter_exp_max'] = exp_max
        session['search_keyword'] = search_keyword
        session['filter_skill'] = filter_skill
        session['filter_role'] = filter_role
        
        flash('Filters applied successfully!', 'success')
        
    except Exception as e:
        flash(f'Error applying filters: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/download/<format>')
def download_results(format):
    """Download filtered results in specified format"""
    try:
        if not session.get('results'):
            flash('No results available for download.', 'error')
            return redirect(url_for('index'))
        
        # Apply filters to get current filtered results
        filtered_results = get_filtered_results()
        
        if not filtered_results:
            flash('No results match the current filters.', 'warning')
            return redirect(url_for('index'))
        
        # Create DataFrame
        df = pd.DataFrame(filtered_results)
        
        if format == 'csv':
            output = StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return send_file(
                BytesIO(output.getvalue().encode()),
                mimetype='text/csv',
                as_attachment=True,
                download_name='filtered_resumes.csv'
            )
        
        elif format == 'excel':
            output = BytesIO()
            df.to_excel(output, index=False)
            output.seek(0)
            
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='filtered_resumes.xlsx'
            )
        
        elif format == 'pdf':
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            output = BytesIO()
            c = canvas.Canvas(output, pagesize=letter)
            c.drawString(100, 750, "Filtered Resume Results")
            
            y = 720
            for _, row in df.iterrows():
                if y < 50:  # Start new page if needed
                    c.showPage()
                    y = 720
                c.drawString(100, y, f"{row['Filename']}: Score {row['Score']}")
                y -= 20
            
            c.save()
            output.seek(0)
            
            return send_file(
                output,
                mimetype='application/pdf',
                as_attachment=True,
                download_name='filtered_resumes.pdf'
            )
        
        else:
            flash('Invalid download format.', 'error')
            return redirect(url_for('index'))
    
    except Exception as e:
        flash(f'Error generating download: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback"""
    try:
        rating = int(request.form.get('rating', 3))
        comments = request.form.get('comments', '').strip()
        
        # Get current filtered results
        filtered_results = get_filtered_results()
        
        feedback_data = {
            "jd": session.get('jd_text', ''),
            "shortlisted": filtered_results,
            "threshold": session.get('threshold', 0.5),
            "rating": rating,
            "comments": comments
        }
        
        save_feedback(feedback_data)
        flash('Feedback submitted successfully!', 'success')
        
    except Exception as e:
        flash(f'Error submitting feedback: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/get_filter_options')
def get_filter_options():
    """Get available skills and roles for filter dropdowns"""
    try:
        if not session.get('results'):
            return jsonify({'skills': [], 'roles': []})
        
        # Apply threshold filter first
        threshold = session.get('threshold', 0.5)
        shortlisted = [r for r in session['results'] if r['Score'] >= threshold]
        
        all_skills = set()
        all_roles = set()
        
        for result in shortlisted:
            all_skills.update(result.get('All_Skills', []))
            all_roles.update(result.get('All_Roles', []))
        
        return jsonify({
            'skills': sorted(list(all_skills)),
            'roles': sorted(list(all_roles))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

def get_filtered_results():
    """Apply filters to results and return filtered list"""
    if not session.get('results'):
        return []
    
    results = session['results']
    
    # Apply threshold filter
    threshold = session.get('threshold', 0.5)
    filtered = [r for r in results if r['Score'] >= threshold]
    
    # Apply experience filter
    exp_min = session.get('filter_exp_min', 0)
    exp_max = session.get('filter_exp_max', 10)
    filtered = [r for r in filtered if exp_min <= r.get('Experience', 0) <= exp_max]
    
    # Apply keyword search
    search_keyword = session.get('search_keyword', '').strip().lower()
    if search_keyword:
        filtered = [r for r in filtered if 
                   search_keyword in r.get('Skills', '').lower() or
                   search_keyword in r.get('Roles', '').lower() or
                   search_keyword in r.get('Education', '').lower()]
    
    # Apply skill filter
    filter_skill = session.get('filter_skill', 'All')
    if filter_skill != 'All':
        filtered = [r for r in filtered if filter_skill in r.get('All_Skills', [])]
    
    # Apply role filter
    filter_role = session.get('filter_role', 'All')
    if filter_role != 'All':
        filtered = [r for r in filtered if filter_role in r.get('All_Roles', [])]
    
    return filtered

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)