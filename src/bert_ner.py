from transformers import pipeline
from preprocessing import extract_entities
from config import SPACY_MODEL
import spacy

# Load SpaCy
nlp = spacy.load(SPACY_MODEL)

# Load Hugging Face NER pipeline (for better entity recognition)
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

def enhanced_ner(text):
    """
    Use BERT-based NER for better extraction.
    """
    entities = ner_pipeline(text)
    return entities

def compare_ner_methods(text):
    """
    Compare SpaCy and BERT NER.
    """
    spacy_entities = extract_entities(text)
    bert_entities = enhanced_ner(text)
    return {
        "spacy": spacy_entities,
        "bert": bert_entities
    }

if __name__ == "__main__":
    sample = "John Doe works at Google as a Python Developer."
    comparison = compare_ner_methods(sample)
    print("NER Comparison:", comparison)
