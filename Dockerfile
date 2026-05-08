# Use a small, official Python base image (Python 3.11 as required)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies (no cache to keep the image small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app and model artifacts into the image
COPY app.py .
COPY model/ ./model/

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app on container start
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

