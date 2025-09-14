from preprocessing import preprocess_text, extract_entities
from enhanced_skill_mapping import EnhancedSkillExtractor
import re
from collections import Counter

# Initialize enhanced skill extractor
_skill_extractor = EnhancedSkillExtractor()

# Legacy skills list (kept for compatibility)
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
    Extract skills from text using enhanced pattern matching and normalization.
    """
    # Use enhanced skill extraction
    enhanced_skills = _skill_extractor.extract_enhanced_skills(text)
    
    # Fallback to legacy method for additional coverage
    text_lower = text.lower()
    legacy_skills = []
    for skill in SKILLS_LIST:
        if skill in text_lower:
            # Normalize using the enhanced mapper
            normalized = _skill_extractor.mapper.normalize_skill(skill)
            legacy_skills.append(normalized)
    
    # Combine and deduplicate
    all_skills = list(set(enhanced_skills + legacy_skills))
    return all_skills

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
    from dateutil import parser as date_parser
    from datetime import datetime
    patterns = [
        r'(\d+)\s*(?:year|yr)s?\s*(?:of\s*)?experience',
        r'experience\s*(?:of\s*)?(\d+)\s*(?:year|yr)s?',
        r'(\d+)\+?\s*(?:year|yr)s?\s*(?:of\s*)?experience',
        r'(over|more than|at least)\s*(\d+)\s*(?:year|yr)s?\s*(?:of\s*)?experience',
        r'(\d{4})\s*[-–to]+\s*(present|current|\d{4})',
        r'(since)\s*(\d{4})'
    ]
    # 1. Direct years of experience
    for pattern in patterns[:3]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    # 2. Phrases like 'over 10 years'
    match = re.search(patterns[3], text, re.IGNORECASE)
    if match:
        return int(match.group(2))
    # 3. Date ranges (e.g., 2015-2022, Jan 2018 - Present)
    date_range_pattern = r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\s*\d{4})\s*[-–to]+\s*(present|current|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\s*\d{4})'
    matches = re.findall(date_range_pattern, text, re.IGNORECASE)
    periods = []
    now = datetime.now()
    for start_str, end_str in matches:
        try:
            start = date_parser.parse(start_str, default=datetime(1900, 1, 1))
            if end_str.lower() in ['present', 'current']:
                end = now
            else:
                end = date_parser.parse(end_str, default=datetime(1900, 1, 1))
            periods.append((start, end))
        except Exception:
            continue
    # 4. 'Since 2012' style
    match = re.search(patterns[5], text, re.IGNORECASE)
    if match:
        try:
            start = date_parser.parse(match.group(2), default=datetime(1900, 1, 1))
            periods.append((start, now))
        except Exception:
            pass
    # Remove overlapping periods
    periods.sort()
    merged = []
    for start, end in periods:
        if not merged:
            merged.append([start, end])
        else:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1][1] = max(last_end, end)
            else:
                merged.append([start, end])
    total_months = 0
    for start, end in merged:
        months = (end.year - start.year) * 12 + (end.month - start.month)
        if months > 0:
            total_months += months
    if total_months > 0:
        return round(total_months / 12, 1)
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
