from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import re
import pickle
import numpy as np
import PyPDF2
from io import BytesIO

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'your_super_secret_key_replace_with_a_strong_one' # IMPORTANT: Change this!

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirects unauthenticated users to the login page

# Dummy User Management (for demonstration purposes only - USE A DATABASE IN PRODUCTION)
users = {
    "testuser": {
        "password_hash": generate_password_hash("123"), # Hash the password
        "id": "1"
    }
}

class User(UserMixin):
    def __init__(self, id):
        self.id = id

    def get_id(self):
        return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    # This function reloads the user object from the user ID stored in the session
    for username, user_data in users.items():
        if user_data['id'] == user_id:
            return User(user_id)
    return None

# Load Models
try:
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open('models/classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
except FileNotFoundError:
    print("Error: Model files not found. Make sure 'models/' directory and its contents exist.")
    exit() # Exit if models are not available

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

    # Note: The original extract_features also returned a 'score' which isn't used
    # directly by the model. The resume_score is calculated separately.
    # We'll just return the processed features needed for prediction.
    tfidf = vectorizer.transform([cleaned]).toarray()
    extra_features = np.array([[kw_matches, length, has_education]])
    full_features = np.hstack([tfidf, extra_features])
    scaled = scaler.transform(full_features)
    reduced = pca.transform(scaled)

    return reduced

# Calculate Resume Quality Score
def calculate_resume_quality(resume_text):
    words = resume_text.split()
    resume_length = len(words)
    
    # Length Score
    if 300 <= resume_length <= 700:
        length_score = 1.0
    elif resume_length < 300:
        length_score = max(0, 1 - (300 - resume_length) / 300)
    else:
        length_score = max(0, 1 - (resume_length - 700) / 300)

    # Keyword Match Score (using defined keywords)
    cleaned_text = clean_text(resume_text)
    matched_keywords_count = sum([1 for word in keywords if word in cleaned_text])
    keyword_match_score = (matched_keywords_count / len(keywords)) if len(keywords) > 0 else 0

    # Vocabulary Diversity
    vocabulary_diversity = len(set(words)) / len(words) if len(words) > 0 else 0

    # Combine scores into a final quality score (weighted average)
    quality_score = (0.35 * length_score * 100 + 0.35 * keyword_match_score * 100 + 0.3 * vocabulary_diversity * 100)
    return round(quality_score, 2)

# Extract Text from PDF
def extract_text_from_pdf(file_stream):
    reader = PyPDF2.PdfReader(file_stream)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

# Flask Route to Handle Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index')) # If already logged in, redirect to home

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_data = users.get(username)

        if user_data and check_password_hash(user_data['password_hash'], password):
            user = User(user_data['id'])
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

# Flask Route to Handle Logout
@app.route('/logout')
@login_required # Only allow logged-in users to logout
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Flask Route to Handle File Upload and Prediction (main page)
@app.route('/', methods=['GET', 'POST'])
@login_required # Protect the main page, requiring login
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
                try:
                    original_text = extract_text_from_pdf(BytesIO(uploaded_file.read()))
                except Exception as e:
                    flash(f"Error processing PDF: {e}. Please ensure it's a valid PDF.", 'danger')
                    original_text = None
            else:
                flash("Unsupported file format. Please upload a .pdf or .txt file.", 'warning')
                original_text = None

            if original_text and original_text != "Unsupported file format.":
                # Preprocess text, extract features
                features = extract_features(original_text) # No longer returning score from here
                # Predict job role
                role = classifier.predict(features)[0]
    
                # Calculate Resume Quality Score
                resume_score = calculate_resume_quality(original_text)
                predicted_role = role
            elif not original_text: # Handle cases where text extraction failed or format was unsupported
                pass # Flash messages are already handled above
        else:
            flash("No file uploaded. Please select a resume file.", 'warning')

    return render_template(
        'index.html',
        resume_score=(resume_score),
        predicted_role=predicted_role,
        original_text=original_text
    )

# New Route for About page
@app.route('/about')
def about():
    return render_template('about.html')

# New Route for Dashboard page
@app.route('/dashboard')
@login_required # Protect the dashboard page
def dashboard():
    # In a real app, you would fetch user-specific data from a database here.
    # For demonstration, let's use a dummy list.
    user_analyses = [
        {"id": 1, "date": "2024-01-15", "role": "Data Scientist", "score": 85},
        {"id": 2, "date": "2024-02-20", "role": "Machine Learning Engineer", "score": 78}
    ]
    return render_template('dashboard.html', analyses=user_analyses)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True) # Set debug=True for development