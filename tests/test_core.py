import pytest
import sys
sys.path.append('.')
from src.preprocessing import preprocess_text, tokenize_text
from src.nlp_features import extract_skills
from src.matching_engine import compute_tfidf_similarity

def test_preprocess_text():
    text = "Hello World! This is a test."
    processed = preprocess_text(text)
    assert "hello" in processed
    assert "world" in processed
    assert "!" not in processed

def test_tokenize_text():
    text = "Hello world"
    tokens = tokenize_text(text)
    assert tokens == ["hello", "world"]

def test_extract_skills():
    text = "I know Python and Java."
    skills = extract_skills(text)
    assert "python" in skills
    assert "java" in skills

def test_tfidf_similarity():
    sim = compute_tfidf_similarity("Python developer", "Java developer")
    assert 0 <= sim <= 1
