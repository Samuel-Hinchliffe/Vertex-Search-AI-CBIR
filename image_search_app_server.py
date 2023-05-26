"""
This Python file implements a Flask web application for content-based image retrieval (CBIR) using feature extraction. 
The app allows users to upload an image and retrieve similar images based on their visual features.

Dependencies:
- numpy (imported as np)
- PIL (imported as Image)
- classes.FeatureExtractor (imported as FeatureExtractor)
- Flask from flask
- Path from pathlib

Usage:
1. Start the Flask app by running this file.
2. Access the app through the browser at http://localhost:5000/.
3. Upload an image using the provided form.
4. The app will extract features from the query image and compare them to precomputed features of a database of images.
5. The top 30 most similar images will be displayed with their corresponding similarity scores.

Folder Structure:
- The 'static/feature' folder contains precomputed feature files in .npy format.
- The 'static/img' folder contains the corresponding image files.
- The 'static/uploaded' folder is used to store uploaded query images temporarily.

Routes:
- GET '/': Renders the main index.html page with the option to upload an image.
- POST '/': Handles the uploaded image, runs the image search, and renders the index.html page with the query image and search results.

Note: Ensure that the necessary dependencies and folder structure are set up before running the application.
"""
__author__ = "Samuel Hinchliffe"
__version__ = "0.0.1"


import numpy as np
from PIL import Image
from classes.FeatureExtractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Initialize the FeatureExtractor, the features array that will
# contain the features and the paths for all of our images.
extractor = FeatureExtractor()
features = []
image_paths = []
white_list = [".jpg", ".jpeg", ".png", ".webp"]

cache_directory = Path("./static/cache")
dataset_directory = Path("./static/dataset")

for cached_feature_path in cache_directory.glob("*.npy"):
    
    # Add the precomputed feature to the features array
    # for later use in Euclidean distance calculations.
    features.append(np.load(cached_feature_path))

    # Load the image that matches the feature
    img_path = None
    stem = cached_feature_path.stem

    for ext in white_list:
        img_file_path = dataset_directory / (stem + ext)
        if img_file_path.is_file():
            img_path = img_file_path
            break

    # If we have a valid image path, add it to the image_paths array
    if img_path:
        image_paths.append(img_path)

# Convert the features array into a numpy array. 
features = np.array(features)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handle the index route for the web application.
    """
    if request.method == "POST":
        # Get the uploaded image
        file = request.files["uploaded_image"]

        # Save the query image
        img = Image.open(file.stream)
        upload_path = f"static/userUploads/{datetime.now().isoformat().replace(':', '.')}_{file.filename}"
        img.save(upload_path)

        # Run the search
        # Extract features from the query image
        # More about feature extraction: https://en.wikipedia.org/wiki/Feature_extraction
        query = extractor.extract(img)  

        # Calculate the Euclidean distances between the query features and all the stored features
        # More about Euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance
        dists = np.linalg.norm(features - query, axis=1)  

        # Sort the distances and retrieve the indices of the closest matches
         # More about argsort: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
        ids = np.argsort(dists)[:20] 

        # Create a list of (distance, image_path) tuples for the top matching images.
        # Because we wish to display this information on the UI. 
        scores = [(dists[id], image_paths[id]) for id in ids]

        # The 'scores' list now contains the top matching images along with their corresponding distances
        return render_template("index.html", query_path=upload_path, scores=scores)
    return render_template("index.html")

if __name__ == "__main__":
    app.run("0.0.0.0")
