import json
import os
from config import DATA_DIR

def save_feedback(feedback_data):
    feedback_file = os.path.join(DATA_DIR, 'feedback.json')
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            feedbacks = json.load(f)
    else:
        feedbacks = []
    
    feedbacks.append(feedback_data)
    with open(feedback_file, 'w') as f:
        json.dump(feedbacks, f, indent=4)
    print("Feedback saved.")

def load_feedback():
    feedback_file = os.path.join(DATA_DIR, 'feedback.json')
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            return json.load(f)
    return []

# Example usage
if __name__ == "__main__":
    save_feedback({"jd": "Python dev", "top_resume": "Candidate1.docx", "rating": 4})
    print(load_feedback())
