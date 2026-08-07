# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Set the working directory in the container
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install system dependencies required by some ML libraries (like OpenCV/Pillow if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy local code to the container image.
COPY . ./

# Install python dependencies.
# We also explicitly install gunicorn to serve the Django app.
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Run the web service on container startup. 
# Cloud Run sets the PORT environment variable (default 8080).
# Using 1 worker and 8 threads is recommended for Cloud Run's CPU=1 allocation.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 agribot_project.wsgi:application
