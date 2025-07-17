

#  Flask CI/CD Pipeline using Jenkins, Docker & SonarQube

This repository demonstrates a complete DevOps pipeline for a Python Flask application. The project integrates continuous integration and deployment using Jenkins, SonarQube for code quality, Docker for containerization, Trivy for vulnerability scanning, and MongoDB as the backend database. OAuth 2.0 login is implemented via Google for secure authentication.

---

## 📦 Tech Stack

* **Frontend:** HTML, CSS, Bootstrap
* **Backend:** Python Flask
* **Database:** MongoDB
* **CI/CD Tools:** Jenkins, SonarQube, Trivy, Docker
* **Security:** OAuth 2.0 (Google Login), Jenkins credentials

---

## 🔧 CI/CD Pipeline Overview

1. **Code Quality Checks**

   * Pylint + SonarQube Static Analysis
   * Trivy Security Scanning

2. **Build Stage**

   * Docker builds Flask app into an image
   * Docker Hub push (optional)

3. **Deploy Stage**

   * App is deployed in a Docker container
   * Uses secure credentials via Jenkins

---

## 🛠️ Run Locally

1. **Clone the Repo**

   ```bash
   git clone https://github.com/sakhij/devops_sem6.git
   cd devops_sem6
   ```

2. **Create Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**

   * Create a `.env` file and add:

     ```
     MONGO_URI=your_mongodb_uri
     CLIENT_ID=your_google_client_id
     CLIENT_SECRET=your_google_client_secret
     ```

4. **Run App Locally**

   ```bash
   flask run
   ```

---

## 🐳 Docker Usage

1. **Build Docker Image**

   ```bash
   docker build -t flaskapp-ci-cd .
   ```

2. **Run Container**

   ```bash
   docker run -p 5000:5000 flaskapp-ci-cd
   ```

---

## 🔐 Google OAuth Setup

* Go to [Google Developer Console](https://console.developers.google.com)
* Create credentials for an OAuth 2.0 Web App
* Add redirect URI: `http://localhost:5000/callback`

---

## ⚙️ Jenkins Pipeline Setup

* Jenkinsfile is included to automate:

  * Pylint quality check
  * SonarQube scan
  * Docker build & run
  * Secure environment variable injection
  * Optional Slack notification

---

## 🧪 Trivy Security Scanning

To scan your Docker image for vulnerabilities:

```bash
trivy image flaskapp-ci-cd
```

---

## 📸 Sample Screenshot

<img width="1916" height="973" alt="image" src="https://github.com/user-attachments/assets/d9183d10-4b8d-481c-8b9e-f9c3ebca3070" />

