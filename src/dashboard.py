import streamlit as st
import pandas as pd
from job_description import process_job_description
from matching_engine import rank_resumes
from config import DATA_DIR
import json

st.title("Resume Ranking Dashboard")

# Load data
@st.cache_data
def load_data():
    with open(f"{DATA_DIR}/processed/resumes_processed.json", 'r') as f:
        resumes = json.load(f)
    with open(f"{DATA_DIR}/processed/ranked_resumes.json", 'r') as f:
        ranked = json.load(f)
    return resumes, ranked

resumes, ranked = load_data()

# Filters
st.sidebar.header("Filters")
min_score = st.sidebar.slider("Minimum Score", 0.0, 1.0, 0.5)
top_n = st.sidebar.slider("Top N", 5, 50, 10)

filtered_ranked = [r for r in ranked if r['score'] >= min_score][:top_n]

st.subheader(f"Filtered Top {len(filtered_ranked)} Resumes")
df = pd.DataFrame(filtered_ranked)
st.dataframe(df)

# Export
st.download_button("Export to CSV", df.to_csv(index=False), "ranked.csv")
