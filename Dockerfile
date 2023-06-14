# Use the official Python image as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files to the working directory
COPY . .

# Expose port 80 for the Flask web server
EXPOSE 80

# Set the environment variable for Flask
ENV FLASK_APP=image_search_app_server.py

# Start the Flask web server when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]