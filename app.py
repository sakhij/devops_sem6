import os
import re
import pickle
import secrets
from datetime import datetime, timezone
from io import BytesIO
from urllib.parse import urlencode

import numpy as np
import PyPDF2
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from pymongo import MongoClient

load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'fallback-secret-key-change-in-production')

GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
GOOGLE_REDIRECT_URI =  os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:5000/auth/callback')

MONGODB_URI = os.getenv("MONGODB_URI")
DB_CLIENT, DB = None, None

try:
    if not MONGODB_URI:
        raise ValueError("No Atlas URI provided")

    DB_CLIENT = MongoClient(MONGODB_URI, tls=True)
    DB_CLIENT.admin.command('ping')
    DB = DB_CLIENT.resume_analyzer
    print("Connected to MongoDB Atlas")
except Exception as atlas_error:
    print(f" Failed to connect to Atlas: {atlas_error}")

if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    print("WARNING: Google OAuth credentials not found.")

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    """User model for Flask-Login"""
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
    """Load user from DB by Google ID"""
    if DB is None:
        return None
    user_data = DB.users.find_one({"google_id": user_id})
    if user_data:
        return User(
            google_id=user_data['google_id'],
            email=user_data['email'],
            name=user_data['name'],
            picture=user_data.get('picture'),
            _id=user_data['_id']
        )
    return None

# Profile Management Functions
def get_user_profile(user_id):
    """Get extended user profile information"""
    if DB is None:
        return {}
    
    try:
        profile = DB.user_profiles.find_one({"user_id": user_id})
        if not profile:
            # Create default profile
            default_profile = {
                "user_id": user_id,
                "phone": "",
                "location": "",
                "profession": "",
                "email_notifications": True,
                "marketing_emails": False,
                "timezone": "UTC",
                "created_at": datetime.utcnow()
            }
            DB.user_profiles.insert_one(default_profile)
            return default_profile
        return profile
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        return {}

def update_user_profile(user_id, profile_data):
    """Update user profile information"""
    if DB is None:
        return False
    
    try:
        profile_data["updated_at"] = datetime.utcnow()
        result = DB.user_profiles.update_one(
            {"user_id": user_id},
            {"$set": profile_data},
            upsert=True
        )
        return result.modified_count > 0 or result.upserted_id is not None
    except Exception as e:
        print(f"Error updating user profile: {e}")
        return False

def get_user_statistics(user_id):
    """Get user account statistics"""
    if DB is None:
        return {}
    
    try:
        # Get user creation date
        user = DB.users.find_one({"google_id": user_id})
        if not user:
            return {}
        
        # Get analysis count and average score
        analyses = list(DB.analyses.find({"user_id": user_id}))
        analysis_count = len(analyses)
        
        # Calculate average score
        avg_score = None
        if analyses:
            total_score = sum(analysis.get('score', 0) for analysis in analyses)
            avg_score = round(total_score / analysis_count, 1)
        
        # Get current UTC time
        now_utc = datetime.utcnow()
        
        # Calculate days since joining
        created_at = user.get('created_at', now_utc)
        # Ensure created_at is timezone-naive
        if hasattr(created_at, 'tzinfo') and created_at.tzinfo:
            created_at = created_at.replace(tzinfo=None)
        days_since_join = max(0, (now_utc - created_at).days)
        
        # Calculate days since last analysis - FIXED VERSION
        last_analysis_days = 0
        if analyses:
            last_analysis = max(analyses, key=lambda x: x.get('date', datetime.min))
            last_analysis_date = last_analysis.get('date', now_utc)
            
            # Ensure last_analysis_date is timezone-naive
            if hasattr(last_analysis_date, 'tzinfo') and last_analysis_date.tzinfo:
                last_analysis_date = last_analysis_date.replace(tzinfo=None)
            
            # Calculate difference and ensure it's not negative
            time_diff = now_utc - last_analysis_date
            last_analysis_days = max(0, time_diff.days)
        
        return {
            'analysis_count': analysis_count,
            'avg_score': avg_score,
            'days_since_join': days_since_join,
            'last_analysis_days': last_analysis_days,
            'account_created': created_at.strftime('%B %d, %Y'),
            'last_login': user.get('last_login', now_utc).strftime('%B %d, %Y at %I:%M %p')
        }
    except Exception as e:
        print(f"Error fetching user statistics: {e}")
        return {}

def delete_user_data(user_id):
    """Delete all user data but keep account"""
    if DB is None:
        return False
    
    try:
        # Delete analyses
        DB.analyses.delete_many({"user_id": user_id})
        # Delete profile
        DB.user_profiles.delete_one({"user_id": user_id})
        return True
    except Exception as e:
        print(f"Error deleting user data: {e}")
        return False

def delete_user_account(user_id):
    """Delete user account and all associated data"""
    if DB is None:
        return False
    
    try:
        # Delete all user data
        DB.analyses.delete_many({"user_id": user_id})
        DB.user_profiles.delete_one({"user_id": user_id})
        DB.users.delete_one({"google_id": user_id})
        return True
    except Exception as e:
        print(f"Error deleting user account: {e}")
        return False

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
    import sys
    print("Error: Model files not found.")
    sys.exit(1)

KEYWORDS = [
    "python", "data", "machine learning", "statistics", "model", "sql",
    "pandas", "numpy", "regression", "classification", "clustering"
]

def clean_text(text):
    """Cleans input text by removing non-alphabetic characters"""
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def keyword_score(text):
    """Counts number of keyword matches in text"""
    return sum(1 for word in KEYWORDS if word in text)

def extract_features(text):
    """Transforms resume text into model input features"""
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
    """Evaluates quality of resume text"""
    words = resume_text.split()
    resume_length = len(words)
    length_score = 1.0 if 300 <= resume_length <= 700 else max(0, 1 - abs(resume_length - 500) / 300)
    cleaned_text = clean_text(resume_text)
    matched_keywords_count = sum(1 for word in KEYWORDS if word in cleaned_text)
    keyword_match_score = matched_keywords_count / len(KEYWORDS)
    vocabulary_diversity = len(set(words)) / len(words) if words else 0
    quality_score = (
        0.35 * length_score * 100 +
        0.35 * keyword_match_score * 100 +
        0.3 * vocabulary_diversity * 100
    )
    return round(quality_score, 2)

def extract_text_from_pdf(file_stream):
    """Extracts text content from a PDF file stream"""
    reader = PyPDF2.PdfReader(file_stream)
    return ''.join(page.extract_text() or '' for page in reader.pages)

def extract_uploaded_resume_text(uploaded_file):
    """Reads and extracts text from uploaded file"""
    if not uploaded_file:
        flash("No file uploaded. Please select a resume file.", 'warning')
        return None

    filename = uploaded_file.filename.lower()
    try:
        if filename.endswith('.txt'):
            return uploaded_file.read().decode('utf-8', errors='ignore')
        if filename.endswith('.pdf'):
            return extract_text_from_pdf(BytesIO(uploaded_file.read()))
        flash("Unsupported file format. Please upload a .pdf or .txt file.", 'warning')
    except Exception as e:
        flash(f"Error processing file: {e}", 'danger')
    return None

def save_analysis(user_id, predicted_role, score, resume_text=None):
    """Saves analysis to MongoDB"""
    if DB is None:
        return None
    try:
        analysis_data = {
            "user_id": user_id,
            "predicted_role": predicted_role,
            "score": score,
            "date": datetime.utcnow(),  # Use UTC consistently
            "resume_text": resume_text[:1000] if resume_text else None
        }
        result = DB.analyses.insert_one(analysis_data)
        return result.inserted_id
    except Exception as e:
        print(f"Error saving analysis: {e}")
        return None

def get_user_analyses(user_id):
    """Retrieves past resume analyses"""
    if DB is None:
        return []
    try:
        analyses = list(DB.analyses.find({"user_id": user_id}).sort("date", -1).limit(10))
        for analysis in analyses:
            analysis['_id'] = str(analysis['_id'])
            analysis['date'] = analysis['date'].strftime('%Y-%m-%d')
        return analyses
    except Exception as e:
        print(f"Error fetching analyses: {e}")
        return []

# Routes
@app.route('/login')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/auth/google')
def auth_google():
    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state
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
        if request.args.get('state') != session.get('oauth_state'):
            flash('Invalid state parameter. Please try again.', 'danger')
            return redirect(url_for('login'))
        code = request.args.get('code')
        if not code:
            flash('Authorization failed. Please try again.', 'danger')
            return redirect(url_for('login'))

        token_data = {
            'client_id': GOOGLE_CLIENT_ID,
            'client_secret': GOOGLE_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': GOOGLE_REDIRECT_URI,
        }
        token_response = requests.post('https://oauth2.googleapis.com/token', data=token_data)
        token_info = token_response.json()
        headers = {'Authorization': f"Bearer {token_info['access_token']}"}
        user_info = requests.get('https://www.googleapis.com/oauth2/v1/userinfo', headers=headers).json()

        user_data = {
            'google_id': user_info['id'],
            'email': user_info['email'],
            'name': user_info['name'],
            'picture': user_info.get('picture'),
            'last_login': datetime.utcnow(),  # Use UTC consistently
            'updated_at': datetime.utcnow()
        }
        if DB is not None:
            DB.users.update_one(
                {'google_id': user_info['id']},
                {'$set': user_data, '$setOnInsert': {'created_at': datetime.utcnow()}},
                upsert=True
            )

        login_user(User(
            google_id=user_info['id'],
            email=user_info['email'],
            name=user_info['name'],
            picture=user_info.get('picture')
        ))
        session.pop('oauth_state', None)
        flash(f"Welcome, {user_info['name']}!", 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"Authentication error: {str(e)}", 'danger')
        return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    resume_score, predicted_role, original_text = None, None, None
    if request.method == 'POST':
        original_text = extract_uploaded_resume_text(request.files.get('resume'))
        if original_text:
            features = extract_features(original_text)
            predicted_role = classifier.predict(features)[0]
            resume_score = calculate_resume_quality(original_text)
            save_analysis(current_user.id, predicted_role, resume_score, original_text)
    return render_template('index.html', resume_score=resume_score, predicted_role=predicted_role, original_text=original_text)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', analyses=get_user_analyses(current_user.id))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        
        if form_type == 'personal_info':
            # Update personal information
            profile_data = {
                'phone': request.form.get('phone', ''),
                'location': request.form.get('location', ''),
                'profession': request.form.get('profession', '')
            }
            
            # Also update user name in main users collection
            if request.form.get('name') != current_user.name:
                if DB is not None:
                    DB.users.update_one(
                        {"google_id": current_user.id},
                        {"$set": {"name": request.form.get('name')}}
                    )
            
            if update_user_profile(current_user.id, profile_data):
                flash('Personal information updated successfully!', 'success')
            else:
                flash('Error updating personal information.', 'danger')
        
        elif form_type == 'preferences':
            # Update preferences
            profile_data = {
                'email_notifications': 'email_notifications' in request.form,
                'marketing_emails': 'marketing_emails' in request.form,
                'timezone': request.form.get('timezone', 'UTC')
            }
            
            if update_user_profile(current_user.id, profile_data):
                flash('Preferences updated successfully!', 'success')
            else:
                flash('Error updating preferences.', 'danger')
        
        return redirect(url_for('profile'))
    
    # GET request - display profile
    user_profile = get_user_profile(current_user.id)
    user_stats = get_user_statistics(current_user.id)
    
    return render_template('profile.html', 
                         user_profile=user_profile,
                         **user_stats)

@app.route('/delete_user_data', endpoint='delete_user_data')
@login_required
def delete_user_data_route():
    if delete_user_data(current_user.id):
        flash('All your data has been deleted successfully.', 'success')
    else:
        flash('Error deleting your data. Please try again.', 'danger')
    return redirect(url_for('profile'))

@app.route('/delete_account')
@login_required
def delete_account():
    user_name = current_user.name
    if delete_user_account(current_user.id):
        logout_user()
        flash(f'Account for {user_name} has been deleted successfully.', 'info')
        return redirect(url_for('login'))
    else:
        flash('Error deleting account. Please try again.', 'danger')
        return redirect(url_for('profile'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)