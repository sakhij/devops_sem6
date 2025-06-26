from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import re
import pickle
import numpy as np
import PyPDF2
from io import BytesIO
import json
import requests
from dotenv import load_dotenv
from urllib.parse import urlencode
import secrets
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'fallback-secret-key-change-in-production')

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
GOOGLE_REDIRECT_URI = 'http://localhost:5000/auth/callback'

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI')
if not MONGODB_URI:
    print("WARNING: MONGODB_URI not found. Please set it in your .env file.")
    print("Example: MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database_name")

# Initialize MongoDB
try:
    client = MongoClient(MONGODB_URI)
    db = client.resume_analyzer  # Database name
    users_collection = db.users
    analyses_collection = db.analyses
    # Test the connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    client = None
    db = None

# Check if OAuth credentials are configured
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    print("WARNING: Google OAuth credentials not found. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables.")
    print("Create a .env file with your Google OAuth credentials for the app to work properly.")

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, google_id, email, name, picture=None, _id=None):
        self.id = google_id
        self.email = email
        self.name = name
        self.picture = picture
        self._id = _id

    def get_id(self):
        return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    if db is None:
        return None
    
    user_data = users_collection.find_one({"google_id": user_id})
    if user_data:
        return User(
            google_id=user_data['google_id'],
            email=user_data['email'],
            name=user_data['name'],
            picture=user_data.get('picture'),
            _id=user_data['_id']
        )
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
    exit()

# Define Keywords for Score Calculation
keywords = [
    "python", "data", "machine learning", "statistics", "model", "sql",
    "pandas", "numpy", "regression", "classification", "clustering"
]

# Preprocess Text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
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

    # Keyword Match Score
    cleaned_text = clean_text(resume_text)
    matched_keywords_count = sum([1 for word in keywords if word in cleaned_text])
    keyword_match_score = (matched_keywords_count / len(keywords)) if len(keywords) > 0 else 0

    # Vocabulary Diversity
    vocabulary_diversity = len(set(words)) / len(words) if len(words) > 0 else 0

    # Combine scores into a final quality score
    quality_score = (0.35 * length_score * 100 + 0.35 * keyword_match_score * 100 + 0.3 * vocabulary_diversity * 100)
    return round(quality_score, 2)

# Extract Text from PDF
def extract_text_from_pdf(file_stream):
    reader = PyPDF2.PdfReader(file_stream)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

# Save analysis to MongoDB
def save_analysis(user_id, predicted_role, score, resume_text=None):
    if db is None:
        return None
    
    analysis_data = {
        "user_id": user_id,
        "predicted_role": predicted_role,
        "score": score,
        "date": datetime.now(),
        "resume_text": resume_text[:1000] if resume_text else None  # Store first 1000 chars for privacy
    }
    
    try:
        result = analyses_collection.insert_one(analysis_data)
        return result.inserted_id
    except Exception as e:
        print(f"Error saving analysis: {e}")
        return None

# Get user's past analyses from MongoDB
def get_user_analyses(user_id):
    if db is None:
        return []
    
    try:
        analyses = list(analyses_collection.find(
            {"user_id": user_id}
        ).sort("date", -1).limit(10))  # Get latest 10 analyses
        
        # Convert ObjectId and datetime to strings for template
        for analysis in analyses:
            analysis['_id'] = str(analysis['_id'])
            analysis['date'] = analysis['date'].strftime('%Y-%m-%d')
        
        return analyses
    except Exception as e:
        print(f"Error fetching analyses: {e}")
        return []

# Google OAuth Routes
@app.route('/login')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    # Show the login page
    return render_template('login.html')

@app.route('/auth/google')
def auth_google():
    # Generate state parameter for security
    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state
    
    # Build Google OAuth URL
    google_auth_url = 'https://accounts.google.com/o/oauth2/auth?' + urlencode({
        'client_id': GOOGLE_CLIENT_ID,
        'response_type': 'code',
        'scope': 'openid email profile',
        'redirect_uri': GOOGLE_REDIRECT_URI,
        'state': state,
    })
    
    return redirect(google_auth_url)

@app.route('/auth/callback')
def auth_callback():
    try:
        # Verify state parameter
        if request.args.get('state') != session.get('oauth_state'):
            flash('Invalid state parameter. Please try again.', 'danger')
            return redirect(url_for('login'))
        
        # Get authorization code
        code = request.args.get('code')
        if not code:
            flash('Authorization failed. Please try again.', 'danger')
            return redirect(url_for('login'))
        
        # Exchange code for access token
        token_data = {
            'client_id': GOOGLE_CLIENT_ID,
            'client_secret': GOOGLE_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': GOOGLE_REDIRECT_URI,
        }
        
        token_response = requests.post('https://oauth2.googleapis.com/token', data=token_data)
        token_response.raise_for_status()
        token_info = token_response.json()
        
        # Get user info using access token
        headers = {'Authorization': f"Bearer {token_info['access_token']}"}
        user_response = requests.get('https://www.googleapis.com/oauth2/v1/userinfo', headers=headers)
        user_response.raise_for_status()
        user_info = user_response.json()
        
        # Extract user data
        google_id = user_info['id']
        email = user_info['email']
        name = user_info['name']
        picture = user_info.get('picture')
        
        # Save/update user in MongoDB
        if db is not None:
            user_data = {
                'google_id': google_id,
                'email': email,
                'name': name,
                'picture': picture,
                'last_login': datetime.now(),
                'updated_at': datetime.now()
            }
            
            # Upsert user (update if exists, insert if not)
            users_collection.update_one(
                {'google_id': google_id},
                {'$set': user_data, '$setOnInsert': {'created_at': datetime.now()}},
                upsert=True
            )
        
        # Create user object and login
        user = User(google_id=google_id, email=email, name=name, picture=picture)
        login_user(user)
        
        # Clean up session
        session.pop('oauth_state', None)
        
        flash(f'Welcome, {name}!', 'success')
        return redirect(url_for('index'))
        
    except requests.exceptions.RequestException as e:
        flash(f'Authentication error: {str(e)}', 'danger')
        return redirect(url_for('login'))
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'danger')
        return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Main application routes
@app.route('/', methods=['GET', 'POST'])
@login_required
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
                # Extract features and predict
                features = extract_features(original_text)
                role = classifier.predict(features)[0]
                resume_score = calculate_resume_quality(original_text)
                predicted_role = role
                
                # Save analysis to MongoDB
                save_analysis(current_user.id, predicted_role, resume_score, original_text)
                
        else:
            flash("No file uploaded. Please select a resume file.", 'warning')

    return render_template(
        'index.html',
        resume_score=resume_score,
        predicted_role=predicted_role,
        original_text=original_text
    )

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's past analyses from MongoDB
    user_analyses = get_user_analyses(current_user.id)
    return render_template('dashboard.html', analyses=user_analyses)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)