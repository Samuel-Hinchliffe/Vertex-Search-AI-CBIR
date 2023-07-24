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
import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
import json
import argparse
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, utility 
from config.milvius_db import MV_DB_CONFIG
from config.qdrant_db import QD_DB_CONFIG

# Firstly, what database do they wish to use?
parser = argparse.ArgumentParser(description='Select database to search from')
parser.add_argument('--db', choices=['local', 'milivus', 'qdrant'],
                        default='qdrant', help='Select the desired database (local, milivus, or qdrant)')
args = parser.parse_args()
db_name = args.db

# Start web server.
app = Flask(__name__)

if (db_name == 'qdrant'):
    db = QdrantClient(QD_DB_CONFIG.host, port=QD_DB_CONFIG.port)

if (db_name == 'milivus'):
    connections.connect(host=MV_DB_CONFIG.host, port=MV_DB_CONFIG.port)
    db = Collection(MV_DB_CONFIG.database)      # Get an existing collection.
    db.load()

# Initialize the FeatureExtractor, the features array that will
# contain the features and the paths for all of our images.
extractor = FeatureExtractor(gpu_mode=True)

if (db_name == 'local'): 
    features = []
    image_paths = []
    white_list = [".jpg", ".jpeg", ".png", ".webp"]

    cache_directory = Path("./static/cache")
    dataset_directory = Path("./static/dataset")

    # Create a dictionary to map feature file names to image file names

    for cached_feature_path in cache_directory.glob("*.npy"):
    
        # Cross reference the cached feature and the loaded image
        # to prevent mismatching vectors and images.
        for ext in white_list:
            img_file_path = dataset_directory / (cached_feature_path.stem + ext)
            if img_file_path.is_file():
                features.append(np.load(cached_feature_path))
                image_paths.append(img_file_path)
                break
        
    # Convert the features array into a numpy array.
    features = np.array(features)

# Custom filter for rounding to 2 decimal places
@app.template_filter('round2')
def round2_filter(value):
    return round(value, 3)


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
        start_time = time.time()
        query = extractor.extract(img)
        print(query)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken for feature extraction: {time_taken} seconds")

        # For Hardcoded vector compare.
        # query = np.load('./static/cache/529961-4020475-large.npy')

        # Calculate the Euclidean distances between the query features and all the stored features
        # More about Euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance
        start_time = time.time()
        
        if (db_name == 'local'):
            dists = np.linalg.norm(features - query, axis=1)
            # Sort the distances and retrieve the indices of the closest matches
            # More about argsort: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
            ids = np.argsort(dists)[:20]

            # Create a list of (distance, image_path) tuples for the top matching images.
            # Because we wish to display this information on the UI.
            scores = [(dists[id], image_paths[id]) for id in ids]

            # Normalise Local
            # TODO: Put into class.
            scores = [
                    {
                        "score": score,
                        "payload": {
                            "url": str(path)
                        },
                        "tags": {
                            "prediction": "N/A",
                            "confidence": "N/A"
                        }
                    }
                    for score, path in scores
                ]

        if (db_name == 'qdrant'): 
            scores = db.search(
                collection_name=QD_DB_CONFIG.database,
                query_vector=query,
                with_vectors=False,
                with_payload=True,
                limit=20,
            )
            
        if (db_name == 'milivus'): 
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 1}
            query = [query]
            scores = db.search(
                    data=query, 
                    anns_field="vector", 
                    param=search_params,
                    limit=20, 
                    expr=None,
                    output_fields=['tags', 'product_image', 'product_url', 'file_name'],
                    consistency_level="Strong",
                )
            
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Time taken for euclidean distance: {time_taken} seconds")
            parsed_results = []
            for hit in scores[0]:
                parsed_results.append({
                    "score": float(hit.distance),
                    "payload": {"url": hit.entity.get('product_image'), "tags": hit.entity.get('tags')},
                })
            scores = parsed_results;
            print(parsed_results)
        
        

        # The 'scores' list now contains the top matching images along with their corresponding distances
        return render_template("index.html", query_path=upload_path, scores=scores)
    return render_template("index.html")

if __name__ == "__main__":
    app.run("0.0.0.0")