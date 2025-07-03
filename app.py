"""
Flask Resume Analyzer Application

A web application that analyzes resumes using machine learning to predict job roles
and calculate resume quality scores. Includes Google OAuth authentication and
MongoDB data storage.
"""

import os
import re
import pickle
import secrets
import sys
from datetime import datetime, timezone
from io import BytesIO
from urllib.parse import urlencode
import pytz
import html
import bleach

import numpy as np
from pypdf import PdfReader
import requests
from dotenv import load_dotenv
from flask import (Flask, render_template, request, redirect, url_for, flash,
                   session)
from flask_login import (LoginManager, UserMixin, login_user, login_required,
                         logout_user, current_user)
from pymongo import MongoClient

load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY',
                                     'fallback-secret-key-change-in-production')

GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
GOOGLE_REDIRECT_URI = os.getenv('GOOGLE_REDIRECT_URI',
                                'http://localhost:5000/auth/callback')

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
        """Return user ID as string for Flask-Login"""
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


def get_current_utc_time():
    """Get current UTC time as timezone-aware datetime"""
    return datetime.now(timezone.utc)


def make_timezone_naive(dt):
    """Convert timezone-aware datetime to naive UTC datetime"""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# Profile Management Functions
def get_user_profile(user_id):
    """Get extended user profile information"""
    if DB is None:
        return {}

    try:
        user_profile = DB.user_profiles.find_one({"user_id": user_id})
        if not user_profile:
            # Create default profile
            default_profile = {
                "user_id": user_id,
                "phone": "",
                "location": "",
                "profession": "",
                "email_notifications": True,
                "marketing_emails": False,
                "timezone": "UTC",
                "created_at": get_current_utc_time()
            }
            DB.user_profiles.insert_one(default_profile)
            return default_profile
        return user_profile
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        return {}


def update_user_profile(user_id, profile_data):
    """Update user profile information"""
    if DB is None:
        return False

    try:
        profile_data["updated_at"] = get_current_utc_time()
        result = DB.user_profiles.update_one(
            {"user_id": user_id},
            {"$set": profile_data},
            upsert=True
        )
        return result.modified_count > 0 or result.upserted_id is not None
    except Exception as e:
        print(f"Error updating user profile: {e}")
        return False


def calculate_time_differences(user, now_utc):
    """Calculate time differences for user statistics"""
    # Calculate days since joining
    created_at = user.get('created_at', now_utc)
    created_at_naive = make_timezone_naive(created_at)
    days_since_join = max(0, (now_utc - created_at_naive).days)
    
    return days_since_join, created_at_naive


def calculate_analysis_stats(analyses, now_utc):
    """Calculate analysis-related statistics"""
    analysis_count = len(analyses)
    avg_score = None
    last_analysis_days = 0
    
    if analyses:
        # Calculate average score
        total_score = sum(analysis.get('score', 0) for analysis in analyses)
        avg_score = round(total_score / analysis_count, 1)
        
        # Calculate days since last analysis
        last_analysis = max(analyses, key=lambda x: x.get('date', datetime.min))
        last_analysis_date = last_analysis.get('date', now_utc)
        last_analysis_naive = make_timezone_naive(last_analysis_date)
        time_diff = now_utc - last_analysis_naive
        last_analysis_days = max(0, time_diff.days)
    
    return analysis_count, avg_score, last_analysis_days


def get_user_statistics(user_id):
    """Get user account statistics"""
    if DB is None:
        return {}

    try:
        # Get user creation date
        user = DB.users.find_one({"google_id": user_id})
        if not user:
            return {}

        # Get user's timezone preference
        user_profile = DB.user_profiles.find_one({"user_id": user_id})
        user_timezone = (user_profile.get('timezone', 'UTC')
                         if user_profile else 'UTC')

        # Get analysis data
        analyses = list(DB.analyses.find({"user_id": user_id}))
        
        # Get current UTC time (timezone-naive for consistency)
        now_utc = make_timezone_naive(get_current_utc_time())

        # Calculate time differences
        days_since_join, created_at_naive = calculate_time_differences(user, now_utc)
        
        # Calculate analysis statistics
        analysis_count, avg_score, last_analysis_days = calculate_analysis_stats(analyses, now_utc)

        # Convert dates to user's timezone for display
        created_at_user_tz = convert_utc_to_timezone(created_at_naive, user_timezone)
        last_login_utc = user.get('last_login', now_utc)
        last_login_naive = make_timezone_naive(last_login_utc)
        last_login_user_tz = convert_utc_to_timezone(last_login_naive, user_timezone)

        return {
            'analysis_count': analysis_count,
            'avg_score': avg_score,
            'days_since_join': days_since_join,
            'last_analysis_days': last_analysis_days,
            'account_created': created_at_user_tz.strftime('%B %d, %Y'),
            'last_login': last_login_user_tz.strftime('%B %d, %Y at %I:%M %p')
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
    print("Error: Model files not found.")
    sys.exit(1)

KEYWORDS = [
    "python", "data", "machine learning", "statistics", "model", "sql",
    "pandas", "numpy", "regression", "classification", "clustering"
]


def clean_text(text):
    """Clean input text by removing non-alphabetic characters"""
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()


def keyword_score(text):
    """Count number of keyword matches in text"""
    return sum(1 for word in KEYWORDS if word in text)


def extract_features(text):
    """Transform resume text into model input features"""
    cleaned = clean_text(text)
    kw_matches = keyword_score(cleaned)
    length = len(cleaned.split())
    has_education = int("bachelor" in cleaned or "master" in cleaned or
                        "phd" in cleaned)
    tfidf = vectorizer.transform([cleaned]).toarray()
    extra_features = np.array([[kw_matches, length, has_education]])
    full_features = np.hstack([tfidf, extra_features])
    scaled = scaler.transform(full_features)
    reduced = pca.transform(scaled)
    return reduced


def calculate_resume_quality(resume_text):
    """Evaluate quality of resume text"""
    words = resume_text.split()
    resume_length = len(words)
    length_score = (1.0 if 300 <= resume_length <= 700
                    else max(0, 1 - abs(resume_length - 500) / 300))
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
    try:
        reader = PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def sanitize_input(text):
    """Sanitize user input to prevent XSS"""
    if not text:
        return ""
    # Remove HTML tags and escape special characters
    cleaned = bleach.clean(text, tags=[], strip=True)
    return html.escape(cleaned)


def extract_uploaded_resume_text(uploaded_file):
    """Reads and extracts text from uploaded file with improved security"""
    if not uploaded_file:
        flash("No file uploaded. Please select a resume file.", 'warning')
        return None

    filename = uploaded_file.filename
    if not filename:
        flash("Invalid file name.", 'warning')
        return None
    
    filename_lower = filename.lower()
    
    # Check file size (limit to 5MB)
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    uploaded_file.seek(0, 2)  # Seek to end
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        flash("File too large. Please upload a file smaller than 5MB.", 'warning')
        return None
    
    try:
        if filename_lower.endswith('.txt'):
            content = uploaded_file.read().decode('utf-8', errors='ignore')
            return sanitize_input(content)
        elif filename_lower.endswith('.pdf'):
            content = extract_text_from_pdf(BytesIO(uploaded_file.read()))
            return sanitize_input(content)
        else:
            flash("Unsupported file format. Please upload a .pdf or .txt file.", 'warning')
            return None
    except Exception as e:
        flash(f"Error processing file: {str(e)}", 'danger')
        return None


def save_analysis(user_id, predicted_role, score, resume_text=None):
    """Save analysis to MongoDB"""
    if DB is None:
        return None
    try:
        analysis_data = {
            "user_id": user_id,
            "predicted_role": predicted_role,
            "score": score,
            "date": get_current_utc_time(),
            "resume_text": resume_text[:1000] if resume_text else None
        }
        result = DB.analyses.insert_one(analysis_data)
        return result.inserted_id
    except Exception as e:
        print(f"Error saving analysis: {e}")
        return None


def get_user_analyses(user_id):
    """Retrieve past resume analyses"""
    if DB is None:
        return []
    try:
        analyses = list(DB.analyses.find({"user_id": user_id})
                        .sort("date", -1).limit(10))
        for analysis in analyses:
            analysis['_id'] = str(analysis['_id'])
            analysis_date = analysis['date']
            if hasattr(analysis_date, 'tzinfo') and analysis_date.tzinfo:
                analysis_date = analysis_date.astimezone(timezone.utc).replace(tzinfo=None)
            analysis['date'] = analysis_date.strftime('%Y-%m-%d')
        return analyses
    except Exception as e:
        print(f"Error fetching analyses: {e}")
        return []


def convert_utc_to_timezone(utc_datetime, target_timezone='UTC'):
    """
    Convert a UTC datetime to a specific timezone

    Args:
        utc_datetime: datetime object (assumed to be in UTC)
        target_timezone: string representing the target timezone
                        (e.g., 'US/Eastern', 'Asia/Kolkata')

    Returns:
        datetime object in the target timezone
    """
    try:
        # Ensure the datetime is timezone-naive (as your code expects)
        if hasattr(utc_datetime, 'tzinfo') and utc_datetime.tzinfo:
            utc_datetime = utc_datetime.replace(tzinfo=None)

        # Create a timezone-aware UTC datetime
        utc_aware = utc_datetime.replace(tzinfo=timezone.utc)

        # Convert to target timezone
        if target_timezone == 'UTC':
            return utc_datetime  # Return as-is for UTC

        target_tz = pytz.timezone(target_timezone)
        converted = utc_aware.astimezone(target_tz)

        # Return timezone-naive datetime (since your code expects this)
        return converted.replace(tzinfo=None)

    except Exception as e:
        print(f"Error converting timezone: {e}")
        # Fallback to original datetime if conversion fails
        return utc_datetime


# Routes
@app.route('/login')
def login():
    """Login page route"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/auth/google')
def auth_google():
    """Initiate Google OAuth authentication"""
    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state
    google_auth_url = ('https://accounts.google.com/o/oauth2/auth?' +
                       urlencode({
                           'client_id': GOOGLE_CLIENT_ID,
                           'response_type': 'code',
                           'scope': 'openid email profile',
                           'redirect_uri': GOOGLE_REDIRECT_URI,
                           'state': state,
                       }))
    return redirect(google_auth_url)


@app.route('/auth/callback')
def auth_callback():
    """Handle Google OAuth callback"""
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
        token_response = requests.post('https://oauth2.googleapis.com/token',
                                       data=token_data)
        token_info = token_response.json()
        headers = {'Authorization': f"Bearer {token_info['access_token']}"}
        user_info = requests.get(
            'https://www.googleapis.com/oauth2/v1/userinfo',
            headers=headers).json()

        current_time = get_current_utc_time()
        user_data = {
            'google_id': user_info['id'],
            'email': user_info['email'],
            'name': user_info['name'],
            'picture': user_info.get('picture'),
            'last_login': current_time,
            'updated_at': current_time
        }
        if DB is not None:
            DB.users.update_one(
                {'google_id': user_info['id']},
                {'$set': user_data,
                 '$setOnInsert': {'created_at': current_time}},
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
    """Logout user"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    """Main application page for resume analysis"""
    resume_score, predicted_role, original_text = None, None, None
    if request.method == 'POST':
        original_text = extract_uploaded_resume_text(request.files.get('resume'))
        if original_text:
            features = extract_features(original_text)
            predicted_role = classifier.predict(features)[0]
            resume_score = calculate_resume_quality(original_text)
            save_analysis(current_user.id, predicted_role, resume_score,
                          original_text)
    return render_template('index.html', resume_score=resume_score,
                           predicted_role=predicted_role,
                           original_text=original_text)


@app.route('/about')
def about():
    """About page route"""
    return render_template('about.html')


@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard showing analysis history"""
    return render_template('dashboard.html',
                           analyses=get_user_analyses(current_user.id))


def handle_personal_info_update():
    """Handle personal information form submission"""
    profile_data = {
        'phone': request.form.get('phone', ''),
        'location': request.form.get('location', ''),
        'profession': request.form.get('profession', '')
    }

    # Also update user name in main users collection
    new_name = request.form.get('name')
    if new_name and new_name != current_user.name and DB is not None:
            DB.users.update_one(
                {"google_id": current_user.id},
                {"$set": {"name": new_name}}
            )

    if update_user_profile(current_user.id, profile_data):
        flash('Personal information updated successfully!', 'success')
    else:
        flash('Error updating personal information.', 'danger')


def handle_preferences_update():
    """Handle preferences form submission"""
    profile_data = {
        'email_notifications': 'email_notifications' in request.form,
        'marketing_emails': 'marketing_emails' in request.form,
        'timezone': request.form.get('timezone', 'UTC')
    }

    if update_user_profile(current_user.id, profile_data):
        flash('Preferences updated successfully!', 'success')
    else:
        flash('Error updating preferences.', 'danger')


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile management page"""
    if request.method == 'POST':
        form_type = request.form.get('form_type')

        if form_type == 'personal_info':
            handle_personal_info_update()
        elif form_type == 'preferences':
            handle_preferences_update()

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
    """Delete user data but keep account"""
    if delete_user_data(current_user.id):
        flash('All your data has been deleted successfully.', 'success')
    else:
        flash('Error deleting your data. Please try again.', 'danger')
    return redirect(url_for('profile'))


@app.route('/delete_account')
@login_required
def delete_account():
    """Delete user account completely"""
    user_name = current_user.name
    if delete_user_account(current_user.id):
        logout_user()
        flash(f'Account for {user_name} has been deleted successfully.',
              'info')
        return redirect(url_for('login'))

    flash('Error deleting account. Please try again.', 'danger')
    return redirect(url_for('profile'))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
