from preprocessing import preprocess_text, extract_entities
import re
from collections import Counter

# Predefined list of common skills (expandable)
SKILLS_LIST = [
    "python", "java", "javascript", "c++", "c#", "sql", "html", "css", "react", "angular", "node.js",
    "machine learning", "ai", "data science", "nlp", "deep learning", "tensorflow", "pytorch", "scikit-learn",
    "business analysis", "project management", "agile", "scrum", "kanban", "waterfall",
    "excel", "word", "powerpoint", "visio", "jira", "confluence",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux", "windows",
    "mysql", "postgresql", "mongodb", "oracle", "redis",
    "rest api", "graphql", "microservices", "ci/cd", "jenkins", "github actions"
]

def extract_skills(text):
    """
    Extract skills from text using keyword matching.
    """
    text_lower = text.lower()
    extracted_skills = []
    for skill in SKILLS_LIST:
        if skill in text_lower:
            extracted_skills.append(skill)
    return list(set(extracted_skills))  # Remove duplicates

def categorize_skills(skills):
    """
    Categorize skills into groups.
    """
    categories = {
        "Programming Languages": ["python", "java", "javascript", "c++", "c#"],
        "Web Technologies": ["html", "css", "react", "angular", "node.js"],
        "Databases": ["sql", "mysql", "postgresql", "mongodb", "oracle", "redis"],
        "ML/AI": ["machine learning", "ai", "data science", "nlp", "deep learning", "tensorflow", "pytorch", "scikit-learn"],
        "Tools": ["excel", "word", "powerpoint", "visio", "jira", "confluence", "git", "docker", "kubernetes"],
        "Cloud": ["aws", "azure", "gcp"],
        "Methodologies": ["agile", "scrum", "kanban", "waterfall", "business analysis", "project management"],
        "Other": []
    }
    
    categorized = {}
    for category, skill_list in categories.items():
        matched = [skill for skill in skills if skill in skill_list]
        if matched:
            categorized[category] = matched
        if category == "Other":
            other_skills = [skill for skill in skills if skill not in sum(categories.values(), [])]
            if other_skills:
                categorized[category] = other_skills
    return categorized

def extract_roles(text):
    """
    Extract job roles using regex patterns.
    """
    roles = []
    role_patterns = [
        r'\b(?:senior|sr\.?|junior|jr\.?)?\s*(?:business analyst|data scientist|software engineer|project manager|developer|architect|consultant|manager)\b'
    ]
    for pattern in role_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        roles.extend(matches)
    return list(set(roles))

def extract_experience(text):
    """
    Extract years of experience using regex.
    """
    import re
    patterns = [
        r'(\d+)\s*(?:year|yr)s?\s*(?:of\s*)?experience',
        r'experience\s*(?:of\s*)?(\d+)\s*(?:year|yr)s?',
        r'(\d+)\+?\s*(?:year|yr)s?\s*(?:of\s*)?experience'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0  # Default if not found

def extract_education(text):
    """
    Extract education levels using regex.
    """
    import re
    education = []
    edu_patterns = [
        r'\b(?:bachelor|master|phd|doctorate|mba|bs|ms|ba|ma)\b'
    ]
    for pattern in edu_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        education.extend(matches)
    return list(set(education))

def advanced_nlp_features(text):
    """
    Combine all extractions.
    """
    skills = extract_skills(text)
    categorized_skills = categorize_skills(skills)
    roles = extract_roles(text)
    education = extract_education(text)
    experience = extract_experience(text)
    return {
        "skills": skills,
        "categorized_skills": categorized_skills,
        "roles": roles,
        "education": education,
        "experience_years": experience
    }

if __name__ == "__main__":
    sample_text = "John is a Senior Python Developer with experience in Machine Learning, AWS, and Agile methodologies. He has a Master's degree."
    features = advanced_nlp_features(sample_text)
    print("Advanced NLP Features:", features)
