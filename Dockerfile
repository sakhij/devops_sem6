FROM python:3.9-slim

# Set workdir using Linux-style path
WORKDIR /app

# Copy requirements file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the rest of the source code
COPY . .

# Default command
CMD ["python", "app.py"]
