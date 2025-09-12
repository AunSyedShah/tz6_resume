# AI Resume Ranker - Web Application
import streamlit as st
import pandas as pd
import sys
sys.path.append('/workspaces/tz6_resume/src')

from job_description import process_job_description
from data_ingestion import process_resume
from preprocessing import preprocess_resume
from matching_engine import combined_score
from advanced_semantic_matching import get_advanced_matcher, AdvancedSemanticMatcher
import os

# Initialize session state
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = ""
if 'jd_option' not in st.session_state:
    st.session_state.jd_option = "Text"
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'uploaded_paths' not in st.session_state:
    st.session_state.uploaded_paths = {}
if 'results' not in st.session_state:
    st.session_state.results = []
if 'evaluation_done' not in st.session_state:
    st.session_state.evaluation_done = False
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5
if 'filter_exp_min' not in st.session_state:
    st.session_state.filter_exp_min = 0
if 'filter_exp_max' not in st.session_state:
    st.session_state.filter_exp_max = 10
if 'search_keyword' not in st.session_state:
    st.session_state.search_keyword = ""
if 'filter_skill' not in st.session_state:
    st.session_state.filter_skill = "All"
if 'filter_role' not in st.session_state:
    st.session_state.filter_role = "All"
if 'jd_data' not in st.session_state:
    st.session_state.jd_data = None
if 'use_advanced_matching' not in st.session_state:
    st.session_state.use_advanced_matching = False
if 'advanced_matcher' not in st.session_state:
    st.session_state.advanced_matcher = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = "all-mpnet-base-v2"
if 'faiss_index_built' not in st.session_state:
    st.session_state.faiss_index_built = False

if 'processed_resumes' not in st.session_state:
    st.session_state.processed_resumes = []

# Page Configuration
st.set_page_config(
    page_title="AI Resume Ranker Pro", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 AI Resume Ranker Pro")
st.markdown("*Advanced AI-powered resume screening with semantic matching and ML ranking*")

# Sidebar Status
with st.sidebar:
    st.header("� System Status")
    
    # Advanced matching status
    if st.session_state.use_advanced_matching:
        st.success(f"🚀 Advanced Matching: {st.session_state.embedding_model}")
        if st.session_state.faiss_index_built:
            st.success("✅ FAISS Index: Ready")
        else:
            st.warning("⚠️ FAISS Index: Not built")
    else:
        st.info("📊 Traditional Matching: Active")
    
    st.divider()

# Status indicator
col1, col2, col3 = st.columns(3)
with col1:
    if st.session_state.jd_text:
        st.success("✅ JD Loaded")
    else:
        st.info("📝 JD: Not loaded")
with col2:
    if st.session_state.uploaded_files:
        st.success(f"✅ {len(st.session_state.uploaded_files)} Resume(s)")
    else:
        st.info("📄 Resumes: Not uploaded")
with col3:
    if st.session_state.evaluation_done:
        st.success("✅ Evaluation Complete")
    else:
        st.info("⚡ Evaluation: Pending")

# JD Input
st.subheader("Job Description")
jd_option = st.radio("Input Method", ["Text", "Upload File"], index=0 if st.session_state.jd_option == "Text" else 1, key="jd_option_radio")
st.session_state.jd_option = jd_option

jd_text = ""
if jd_option == "Text":
    jd_text = st.text_area("Enter Job Description", value=st.session_state.jd_text, height=150, key="jd_text_area")
else:
    jd_file = st.file_uploader("Upload JD (.docx or .txt)", type=["docx", "txt"], key="jd_file_uploader")
    if jd_file:
        if jd_file.type == "text/plain":
            jd_text = str(jd_file.read(), "utf-8")
        elif jd_file.name.endswith(".docx"):
            # Process .docx for JD
            temp_path = f"/tmp/{jd_file.name}"
            with open(temp_path, "wb") as f:
                f.write(jd_file.getbuffer())
            from data_ingestion import extract_text_from_docx
            jd_text = extract_text_from_docx(temp_path)
        st.success("JD loaded successfully!")

st.session_state.jd_text = jd_text

import os
import shutil
from pathlib import Path

# Directory for storing uploaded files
UPLOAD_DIR = Path("/tmp/resume_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Advanced Matching Configuration
st.subheader("⚙️ Matching Configuration")
col1, col2 = st.columns(2)

with col1:
    use_advanced = st.checkbox(
        "🚀 Use Advanced Semantic Matching (FAISS + Transformers)",
        value=st.session_state.use_advanced_matching,
        help="Enable FAISS vector database with state-of-the-art transformer embeddings for better semantic understanding"
    )
    st.session_state.use_advanced_matching = use_advanced

with col2:
    if use_advanced:
        model_options = ["all-mpnet-base-v2", "all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"]
        selected_model = st.selectbox(
            "Embedding Model",
            model_options,
            index=0,
            help="Choose the transformer model for embeddings. all-mpnet-base-v2 provides the best quality."
        )
        st.session_state.embedding_model = selected_model

# Resume Upload
st.subheader("Upload Resumes")
uploaded_files = st.file_uploader("Choose .docx files", type="docx", accept_multiple_files=True, key="resume_uploader")

# Update session state with uploaded files
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
    uploaded_paths = {}
    for uploaded_file in uploaded_files:
        # Save to persistent location
        save_path = UPLOAD_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_paths[uploaded_file.name] = save_path
        st.success(f"Saved {uploaded_file.name}")
    st.session_state.uploaded_paths = uploaded_paths
else:
    uploaded_paths = st.session_state.uploaded_paths

# Show previously uploaded files if they exist
if st.session_state.uploaded_files:
    st.write(f"**{len(st.session_state.uploaded_files)} resume(s) uploaded:**")
    for file in st.session_state.uploaded_files:
        st.write(f"- {file.name}")

if st.button("🚀 Evaluate Resumes", key="evaluate_button"):
    if st.session_state.jd_text.strip() and st.session_state.uploaded_files:
        with st.spinner("Processing JD and evaluating resumes..."):
            # Process JD
            jd_data = process_job_description(st.session_state.jd_text)
            st.session_state.jd_data = jd_data
            
            # Process all resumes
            processed_resumes = []
            for uploaded_file in st.session_state.uploaded_files:
                # Save temporarily
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process resume
                resume_data = process_resume(temp_path)
                resume_data = preprocess_resume(resume_data)
                resume_data['filename'] = uploaded_file.name  # Add filename for tracking
                processed_resumes.append(resume_data)
            
            st.session_state.processed_resumes = processed_resumes
            
            # Initialize Advanced Matcher if using advanced matching
            if st.session_state.use_advanced_matching:
                if st.session_state.advanced_matcher is None:
                    with st.spinner("🔧 Initializing Advanced Semantic Matcher..."):
                        from advanced_semantic_matching import AdvancedSemanticMatcher
                        st.session_state.advanced_matcher = AdvancedSemanticMatcher(
                            model_name=st.session_state.embedding_model
                        )
                        st.success(f"✅ Advanced matcher initialized with {st.session_state.embedding_model}")
                
                # Build FAISS index if not built
                if not st.session_state.faiss_index_built:
                    with st.spinner("🗂️ Building FAISS vector database..."):
                        # Prepare resume data for the advanced matcher
                        resume_data_for_faiss = []
                        for i, resume in enumerate(processed_resumes):
                            # Combine all text content for embedding
                            text_content = f"{resume['raw_text']} {' '.join(resume['features'].get('skills', []))}"
                            resume_data_for_faiss.append({
                                'text': text_content,
                                'filename': resume['filename'],
                                'index': i
                            })
                        
                        st.session_state.advanced_matcher.build_faiss_index(resume_data_for_faiss)
                        st.session_state.faiss_index_built = True
                        st.success("✅ FAISS vector database built successfully!")
            
            # Rank resumes based on selected method
            if st.session_state.use_advanced_matching:
                # Use advanced semantic matching with FAISS
                with st.spinner("🧠 Advanced semantic ranking in progress..."):
                    jd_combined = f"{jd_data['raw_text']} {' '.join(jd_data['features']['skills'])}"
                    
                    # Use batch scoring method
                    scoring_results = st.session_state.advanced_matcher.batch_score_resumes(processed_resumes, jd_combined)
                    
                    # Create results with advanced semantic scores
                    results = []
                    for i, resume in enumerate(processed_resumes):
                        # Get the corresponding score result
                        if i < len(scoring_results):
                            score_result = scoring_results[i]
                            semantic_score = score_result['advanced_scores']['advanced_score'] * 100  # Convert to percentage
                        else:
                            semantic_score = 50.0  # Default fallback score
                        
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
                    
                    # Sort by semantic score
                    results = sorted(results, key=lambda x: x['Score'], reverse=True)
                    st.success(f"✅ Advanced semantic ranking completed using {st.session_state.embedding_model}!")
            else:
                # Use traditional scoring
                with st.spinner("📊 Traditional scoring in progress..."):
                    results = []
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
                    # Sort by score
                    results.sort(key=lambda x: x['Score'], reverse=True)
                    st.success("✅ Traditional scoring completed!")
        
            st.session_state.results = results
            st.session_state.evaluation_done = True
            st.success("✅ Evaluation complete!")
    else:
        st.error("❌ Please enter a JD and upload at least one resume.")

# Display results if evaluation is done
if st.session_state.evaluation_done and st.session_state.results:
        
        # Results Overview
        if st.session_state.results:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Resumes", len(st.session_state.results))
            with col2:
                avg_score = sum(r['Score'] for r in st.session_state.results) / len(st.session_state.results)
                st.metric("Average Score", f"{avg_score:.2f}")
            with col3:
                engine_type = "Advanced Semantic" if st.session_state.use_advanced_matching else "Traditional"
                st.metric("Engine Type", engine_type)
            
            # Advanced Scoring Details (if available)
            if st.session_state.use_advanced_matching and 'Semantic Score' in st.session_state.results[0]:
                with st.expander("🚀 Advanced Semantic Matching Breakdown (Top Resume)"):
                    top_resume = st.session_state.results[0]
                    st.write(f"**Filename:** {top_resume['Filename']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Semantic Score", f"{top_resume['Score']:.2f}%")
                    with col2:
                        st.metric("Model", st.session_state.embedding_model)
                    with col3:
                        st.metric("Method", "FAISS Vector Search")
                    
                    st.write(f"**Skills:** {top_resume['Skills']}")
                    st.write(f"**Roles:** {top_resume['Roles']}")
                    st.write(f"**Education:** {top_resume['Education']}")
                    st.write(f"**Experience:** {top_resume['Experience']} years")
            
            # Debug: Show extracted features for first resume
            with st.expander("🔍 Debug: Raw Features (First Resume)"):
                if st.session_state.processed_resumes:
                    first_processed = st.session_state.processed_resumes[0]
                    st.write("**Raw Features Extracted:**")
                    st.json(first_processed['features'])
        
        # Shortlisting Threshold
        threshold = st.slider("Shortlisting Threshold (Min Score)", 0.0, 1.0, st.session_state.threshold, 0.1, key="threshold_slider")
        st.session_state.threshold = threshold
        shortlisted = [r for r in st.session_state.results if r["Score"] >= threshold]
        
        filter_exp_min = st.slider("Min Experience (Years)", 0, 20, st.session_state.filter_exp_min, key="exp_min_slider")
        filter_exp_max = st.slider("Max Experience (Years)", 0, 20, st.session_state.filter_exp_max, key="exp_max_slider")
        st.session_state.filter_exp_min = filter_exp_min
        st.session_state.filter_exp_max = filter_exp_max
        
        # Additional filters
        search_keyword = st.text_input("Search by keyword (skills, roles, education)", st.session_state.search_keyword, key="search_input")
        st.session_state.search_keyword = search_keyword
        
        # Get unique skills and roles from results for filter options
        all_skills = set()
        all_roles = set()
        for r in shortlisted:
            # Only add actual skills/roles, not placeholder text
            if r["All_Skills"]:
                all_skills.update(r["All_Skills"])
            if r["All_Roles"]:
                all_roles.update(r["All_Roles"])
        
        # Ensure we have at least "All" option
        skill_options = ["All"] + sorted(list(all_skills)) if all_skills else ["All"]
        role_options = ["All"] + sorted(list(all_roles)) if all_roles else ["All"]
        
        filter_skill = st.selectbox("Filter by Skill", skill_options, index=skill_options.index(st.session_state.filter_skill) if st.session_state.filter_skill in skill_options else 0, key="skill_select")
        filter_role = st.selectbox("Filter by Role", role_options, index=role_options.index(st.session_state.filter_role) if st.session_state.filter_role in role_options else 0, key="role_select")
        st.session_state.filter_skill = filter_skill
        st.session_state.filter_role = filter_role
        
        # Semantic Search (FAISS-powered)
        if st.session_state.use_advanced_matching and st.session_state.faiss_index_built:
            st.divider()
            st.subheader("🔍 Semantic Search")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                semantic_query = st.text_input(
                    "Find resumes semantically similar to:", 
                    placeholder="e.g., 'React developer with cloud experience'",
                    key="semantic_search"
                )
            with col2:
                top_k_semantic = st.number_input("Top matches", min_value=1, max_value=20, value=5, key="top_k_semantic")
            
            if semantic_query and st.button("🚀 Search", key="semantic_search_btn"):
                with st.spinner("Searching similar resumes..."):
                    search_results = st.session_state.advanced_matcher.semantic_search(semantic_query, top_k=top_k_semantic)
                    
                    if search_results:
                        st.success(f"Found {len(search_results)} semantically similar resumes:")
                        
                        # Display semantic search results
                        for i, result in enumerate(search_results):
                            filename = result['metadata']['filename']
                            similarity = result['score']
                            
                            # Find corresponding resume in results
                            resume_match = next((r for r in st.session_state.results if r['Filename'] == filename), None)
                            if resume_match:
                                with st.expander(f"🎯 {i+1}. {filename} (Similarity: {similarity:.3f})"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Semantic Similarity", f"{similarity:.3f}")
                                    with col2:
                                        st.metric("Overall Score", f"{resume_match['Score']:.3f}")
                                    
                                    st.write(f"**Skills:** {resume_match['Skills']}")
                                    st.write(f"**Roles:** {resume_match['Roles']}")
                    else:
                        st.info("No similar resumes found for your query.")
            
            st.divider()
        
        # Apply filters
        filtered = shortlisted
        if search_keyword:
            # Search in actual content, not placeholder text
            for r in filtered[:]:  # Use slice copy to avoid modification during iteration
                searchable_text = ""
                if r["All_Skills"]:
                    searchable_text += " ".join(r["All_Skills"]).lower() + " "
                if r["All_Roles"]:
                    searchable_text += " ".join(r["All_Roles"]).lower() + " "
                if r["Education"] not in ["No education detected", ""]:
                    searchable_text += r["Education"].lower()
                
                if search_keyword.lower() not in searchable_text:
                    filtered.remove(r)
        
        if filter_skill != "All":
            filtered = [r for r in filtered if r["All_Skills"] and filter_skill in r["All_Skills"]]
        
        if filter_role != "All":
            filtered = [r for r in filtered if r["All_Roles"] and filter_role in r["All_Roles"]]
        
        # Experience filter
        filtered = [r for r in filtered if filter_exp_min <= r.get("Experience", 0) <= filter_exp_max]
        
        # Display results
        st.subheader(f"Filtered Results ({len(filtered)} candidates)")
        if filtered:
            # Get JD skills for highlighting
            jd_skills = st.session_state.jd_data['features']['skills'] if st.session_state.jd_data else []
            
            for idx, candidate in enumerate(filtered):
                score = candidate["Score"]
                skills = candidate["Skills"].split(", ") if candidate["Skills"] != "No skills detected" else []
                highlighted_skills = [f"**{s}**" if s in jd_skills else s for s in skills]  # Bold matching skills
                
                # Score display with method information
                score_text = f"Score: {score}"
                if st.session_state.use_advanced_matching and 'Semantic Score' in candidate:
                    score_text += f" (Semantic: {candidate['Semantic Score']:.2f}%)"
                else:
                    score_text += f" (Traditional)"
                
                with st.expander(f"{idx+1}. {candidate['Filename']} ({score_text})"):
                    # Score breakdown
                    if st.session_state.use_advanced_matching and 'Semantic Score' in candidate:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Final Score", f"{candidate['Score']:.2f}%")
                        with col2:
                            st.metric("Semantic Score", f"{candidate['Semantic Score']:.2f}%")  
                        with col3:
                            st.metric("Method", candidate.get('Method', 'N/A'))
                        st.divider()
                    
                    st.write(f"**Skills:** {', '.join(highlighted_skills) if highlighted_skills else 'No skills detected'}")
                    st.write(f"**Roles:** {candidate['Roles']}")
                    st.write(f"**Education:** {candidate['Education']}")
                    st.write(f"**Experience:** {candidate.get('Experience', 'N/A')} years")
                    
                    # Download original resume
                    file_path = st.session_state.uploaded_paths.get(candidate['Filename'])
                    if file_path and file_path.exists():
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label=f"Download Original {candidate['Filename']}",
                                data=f,
                                file_name=candidate['Filename'],
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key=f"download_{idx}"
                            )
                    else:
                        st.info("Original file not available for download.")
            
            # Table view
            results_df = pd.DataFrame(filtered)
            
            # Select columns to display based on engine type
            if st.session_state.use_advanced_matching and 'Semantic Score' in filtered[0]:
                display_columns = ["Filename", "Score", "Semantic Score", "Method", "Skills", "Roles", "Education", "Experience"]
                table_df = results_df[display_columns].sort_values(by="Score", ascending=False)
            else:
                display_columns = ["Filename", "Score", "Method", "Skills", "Roles", "Education", "Experience"]
                table_df = results_df[display_columns].sort_values(by="Score", ascending=False)
            
            st.subheader("📊 Results Table")
            st.dataframe(table_df, use_container_width=True)
            
            # Download options
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered as CSV",
                data=csv,
                file_name="filtered_resumes.csv",
                mime="text/csv"
            )
            
            # Excel export
            from io import BytesIO
            buffer = BytesIO()
            results_df.to_excel(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="Download as Excel",
                data=buffer,
                file_name="filtered_resumes.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # PDF export (basic)
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.drawString(100, 750, "Filtered Resume Results")
            y = 720
            for _, row in results_df.iterrows():
                c.drawString(100, y, f"{row['Filename']}: Score {row['Score']}")
                y -= 20
            c.save()
            pdf_buffer.seek(0)
            st.download_button(
                label="Download as PDF",
                data=pdf_buffer,
                file_name="filtered_resumes.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("No candidates match the filters.")
        
        # Feedback
        st.subheader("Provide Feedback")
        with st.form("feedback_form"):
            rating = st.slider("Rate the evaluation (1-5)", 1, 5, 3)
            comments = st.text_area("Comments")
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                from feedback import save_feedback
                save_feedback({
                    "jd": st.session_state.jd_text,
                    "shortlisted": shortlisted,
                    "threshold": st.session_state.threshold,
                    "rating": rating,
                    "comments": comments
                })
                st.success("Feedback submitted!")
else:
    st.error("Please enter a JD and upload at least one resume.")

st.sidebar.header("About")
st.sidebar.write("This app evaluates resumes against a job description using NLP and ML.")