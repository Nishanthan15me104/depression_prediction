# 1. Use a slim Python image to keep the size small
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only requirements first (this leverages Docker caching)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the entire src folder and other necessary files
# This includes your new 'src/api/static_models/final_model.pkl'
COPY . .

# 7. Set Environment Variables
# This tells your Python code to use the static model, not MLflow
ENV APP_ENV=PROD
ENV PYTHONPATH=/app

# 8. Expose the port FastAPI runs on
EXPOSE 8000

# 9. Command to run the API
CMD ["python", "run_api.py"]