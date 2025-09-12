import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import spacy
from config import SPACY_MODEL

# Load SpaCy model
nlp = spacy.load(SPACY_MODEL)

# NLTK setup
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def tokenize_text(text):
    """
    Tokenize the input text.
    """
    return word_tokenize(text.lower())

def remove_stopwords(tokens):
    """
    Remove stopwords from tokens.
    """
    return [word for word in tokens if word not in stop_words and word.isalnum()]

def stem_tokens(tokens):
    """
    Apply stemming to tokens.
    """
    return [stemmer.stem(word) for word in tokens]

def lemmatize_tokens(tokens):
    """
    Apply lemmatization to tokens.
    """
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text):
    """
    Full preprocessing pipeline: tokenize, remove stopwords, lemmatize.
    """
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)

def extract_entities(text):
    """
    Use SpaCy NER to extract entities like skills, roles, etc.
    """
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    return entities

from nlp_features import advanced_nlp_features

def preprocess_resume(resume_data):
    """
    Preprocess a single resume dict.
    """
    raw_text = resume_data['raw_text']
    processed_text = preprocess_text(raw_text)
    entities = extract_entities(raw_text)
    features = advanced_nlp_features(raw_text)
    resume_data['processed_text'] = processed_text
    resume_data['entities'] = entities
    resume_data['features'] = features
    return resume_data

if __name__ == "__main__":
    # Test with sample text
    sample_text = "John Doe is a Python developer with experience in machine learning."
    print("Original:", sample_text)
    print("Processed:", preprocess_text(sample_text))
    print("Entities:", extract_entities(sample_text))
