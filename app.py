from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash
from authlib.integrations.flask_client import OAuth
import os
import re
import pickle
import numpy as np
import PyPDF2
from io import BytesIO

# Flask app setup
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'your_super_secret_key_replace_with_a_strong_one'

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# OAuth setup
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://openidconnect.googleapis.com/v1/',
    client_kwargs={'scope': 'openid email profile'}
)

# Dummy user store
users = {}

class User(UserMixin):
    def __init__(self, id):
        self.id = id

    def get_id(self):
        return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    for _, user_data in users.items():
        if user_data['id'] == user_id:
            return User(user_id)
    return None

# Load ML models
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
    print("Model files not found.")
    exit()

# Keywords
keywords = ["python", "data", "machine learning", "statistics", "model", "sql",
            "pandas", "numpy", "regression", "classification", "clustering"]

# Utilities
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def keyword_score(text):
    return sum([1 for word in keywords if word in text])

def extract_features(text):
    cleaned = clean_text(text)
    kw_matches = keyword_score(cleaned)
    length = len(cleaned.split())
    has_education = int("bachelor" in cleaned or "master" in cleaned or "phd" in cleaned)
    tfidf = vectorizer.transform([cleaned]).toarray()
    extra_features = np.array([[kw_matches, length, has_education]])
    full_features = np.hstack([tfidf, extra_features])
    scaled = scaler.transform(full_features)
    reduced = pca.transform(scaled)
    return reduced

def calculate_resume_quality(resume_text):
    words = resume_text.split()
    resume_length = len(words)
    length_score = 1.0 if 300 <= resume_length <= 700 else max(0, 1 - abs(resume_length - 500) / 300)
    cleaned_text = clean_text(resume_text)
    matched_keywords_count = keyword_score(cleaned_text)
    keyword_match_score = matched_keywords_count / len(keywords)
    vocabulary_diversity = len(set(words)) / len(words) if words else 0
    score = (0.35 * length_score + 0.35 * keyword_match_score + 0.3 * vocabulary_diversity) * 100
    return round(score, 2)

def extract_text_from_pdf(file_stream):
    reader = PyPDF2.PdfReader(file_stream)
    return ''.join([page.extract_text() or '' for page in reader.pages])

# Google login
@app.route('/login')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    redirect_uri = url_for('authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    token = google.authorize_access_token()
    userinfo = google.parse_id_token(token)
    if not userinfo:
        flash("Google login failed", "danger")
        return redirect(url_for('login'))
    email = userinfo['email']
    if email not in users:
        users[email] = {'id': str(len(users)+1)}
    user = User(users[email]['id'])
    login_user(user)
    session['email'] = email
    flash(f"Logged in as {email}", "success")
    return redirect(url_for('index'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    resume_score = predicted_role = original_text = None
    if request.method == 'POST':
        uploaded_file = request.files['resume']
        if uploaded_file:
            filename = uploaded_file.filename
            if filename.endswith('.txt'):
                original_text = uploaded_file.read().decode('utf-8', errors='ignore')
            elif filename.endswith('.pdf'):
                original_text = extract_text_from_pdf(BytesIO(uploaded_file.read()))
            else:
                flash("Unsupported file format", "warning")
                return redirect(url_for('index'))
            features = extract_features(original_text)
            role = classifier.predict(features)[0]
            resume_score = calculate_resume_quality(original_text)
            predicted_role = role
    return render_template("index.html", resume_score=resume_score, predicted_role=predicted_role, original_text=original_text)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user_analyses = [
        {"id": 1, "date": "2024-01-15", "role": "Data Scientist", "score": 85},
        {"id": 2, "date": "2024-02-20", "role": "ML Engineer", "score": 78}
    ]
    return render_template('dashboard.html', analyses=user_analyses)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
