# Use stable Python
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 10000

# Start FastAPI server
CMD ["bash", "-c", "cd backend && uvicorn main:app --host 0.0.0.0 --port 10000"]