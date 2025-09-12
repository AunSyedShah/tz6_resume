from preprocessing import preprocess_text, extract_entities
from nlp_features import advanced_nlp_features

def process_job_description(jd_text):
    """
    Process a job description text: preprocess and extract entities.
    """
    processed_jd = preprocess_text(jd_text)
    entities = extract_entities(jd_text)
    features = advanced_nlp_features(jd_text)
    return {
        "raw_text": jd_text,
        "processed_text": processed_jd,
        "entities": entities,
        "features": features
    }

if __name__ == "__main__":
    # Test with sample JD
    sample_jd = "We are looking for a Senior Python Developer with experience in AI and machine learning."
    result = process_job_description(sample_jd)
    print("JD Processed:", result)
