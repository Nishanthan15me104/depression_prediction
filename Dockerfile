# 1. Use a slim Python image
FROM python:3.10-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy ONLY the production requirements (Renamed as requested)
COPY requirements-prod.txt .

# 5. Install Python dependencies from the production file
RUN pip install --no-cache-dir -r requirements-prod.txt

# 6. Copy the entire project
COPY . .

# 7. Set Environment Variables
ENV APP_ENV=PROD
ENV PYTHONPATH=/app

# 8. Expose the port
EXPOSE 8000

# 9. Command to run the API
CMD ["python", "run_api.py"]