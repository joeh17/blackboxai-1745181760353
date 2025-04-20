# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r datalexis/requirements.txt

# Expose port 8000 for the Flask app
EXPOSE 8000

# Define environment variable
ENV FLASK_APP=datalexis/web_app.py

# Run the Flask app
CMD ["python", "datalexis/web_app.py"]
