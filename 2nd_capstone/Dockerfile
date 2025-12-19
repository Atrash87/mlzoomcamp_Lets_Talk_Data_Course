FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and application code
COPY deployment/ /app/deployment/
COPY app.py /app/

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
