from flask import Flask, render_template, request
import os
import re
import pickle
import numpy as np
import PyPDF2
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load Models
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/pca.pkl', 'rb') as f:
    pca = pickle.load(f)
with open('models/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Define Keywords for Score Calculation
keywords = [
    "python", "data", "machine learning", "statistics", "model", "sql",
    "pandas", "numpy", "regression", "classification", "clustering"
]

# Preprocess Text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.lower()

# Keyword Matching
def keyword_score(text):
    return sum([1 for word in keywords if word in text])

# Extract Features
def extract_features(text, max_kw=10, max_len=1000):
    cleaned = clean_text(text)
    kw_matches = keyword_score(cleaned)
    length = len(cleaned.split())
    has_education = int("bachelor" in cleaned or "master" in cleaned or "phd" in cleaned)

    score = (
        (kw_matches / max_kw) * 0.5 +
        (length / max_len) * 0.3 +
        has_education * 0.2
    )

    tfidf = vectorizer.transform([cleaned]).toarray()
    extra_features = np.array([[kw_matches, length, has_education]])
    full_features = np.hstack([tfidf, extra_features])
    scaled = scaler.transform(full_features)
    reduced = pca.transform(scaled)

    return reduced, score

# Calculate Resume Quality Score
def calculate_resume_quality(resume_text):
    words = resume_text.split()
    resume_length = len(words)
    
    if 300 <= resume_length <= 700:
        length_score = 1.0
    elif resume_length < 300:
        length_score = max(0, 1 - (300 - resume_length) / 300)
    else:
        length_score = max(0, 1 - (resume_length - 700) / 300)

    # Placeholder for keyword match score and vocabulary diversity
    keyword_match_score = 0.5  # Placeholder value, should be updated dynamically
    vocabulary_diversity = len(set(words)) / len(words) if len(words) > 0 else 0

    quality_score = (0.35 * length_score * 100 + 0.35 * keyword_match_score * 100 + 0.3 * vocabulary_diversity * 100)
    return round(quality_score, 2)

# Extract Text from PDF
def extract_text_from_pdf(file_stream):
    reader = PyPDF2.PdfReader(file_stream)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

# Flask Route to Handle File Upload and Prediction
@app.route('/', methods=['GET', 'POST'])
def index():
    resume_score = None
    predicted_role = None
    original_text = None

    if request.method == 'POST':
        uploaded_file = request.files['resume']
        if uploaded_file:
            filename = uploaded_file.filename
            # Handle .txt file
            if filename.endswith('.txt'):
                original_text = uploaded_file.read().decode('utf-8', errors='ignore')
            # Handle .pdf file
            elif filename.endswith('.pdf'):
                original_text = extract_text_from_pdf(BytesIO(uploaded_file.read()))
            else:
                original_text = "Unsupported file format."

            if original_text:
                # Preprocess text, extract features
                features, _ = extract_features(original_text)
                # Predict job role
                role = classifier.predict(features)[0]
    
                # Calculate Resume Quality Score
                resume_score = calculate_resume_quality(original_text)
                predicted_role = role

    return render_template(
        'index.html',
        resume_score=(resume_score),
        predicted_role=predicted_role,
        original_text=original_text
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
