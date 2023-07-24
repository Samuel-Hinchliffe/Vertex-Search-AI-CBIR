"""
This script extracts features from a dataset of images using a FeatureExtractor class and saves the extracted features
to corresponding files for later use.

Author: Samuel Hinchliffe
Version: 0.0.1

Dependencies:
    numpy (imported as np): A library for numerical computing in Python.
    PIL (imported as Image): The Python Imaging Library for image processing.
    classes.FeatureExtractor (imported as FeatureExtractor): A custom class for feature extraction.
    pathlib: A module for working with file paths.

Usage:
    Ensure that the necessary dependencies are installed.
    Place the images to extract features from in the 'static/dataset' directory.
    Run this script to extract features from the images and save them to corresponding files.
    The extracted features will be saved in the 'static/cache' directory with the same file names as the original images, but with the '.npy' extension.

Folder Structure:
    The 'static/dataset' directory contains the images to extract features from.
    The 'static/cache' directory stores the extracted features in .npy files.

Main Function:
    The main function, 'main()', iterates over the image files in the 'static/dataset' directory.
    For each image file, it opens the image, extracts features using the FeatureExtractor class, and saves the features to a corresponding file in the 'static/cache' directory.
    Note: Ensure that the necessary dependencies are installed and the dataset images are in the correct directory structure before running the script.
"""

__author__ = "Samuel Hinchliffe"
__version__ = "0.0.1"

from PIL import Image
from classes.FeatureExtractor import FeatureExtractor
from pathlib import Path
import numpy as np
import concurrent.futures
import io
import requests
import sys
import json
import time
import glob
import os
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, utility 
from guppy import hpy

conn = connections.connect(host="127.0.0.1", port=19530)
collection = Collection("image_vectors")
client = QdrantClient("localhost", port=6333)

def process_image(img_url, feature_extractor, img_name, counter):

    
    # Download the image from the given URL
    response = requests.get(img_url)
    image_data = response.content

    # Open the image using PIL from the downloaded data
    image = Image.open(io.BytesIO(image_data))

    # Extract features from the image using the FeatureExtractor class
    feature = feature_extractor.extract(img=image)
    del image
    del image_data
    del response

    # Predictions
    # Not optimised.
    predictions = feature_extractor.predict(img=image)
    
    # labels 
    text_predictions = []
    for predict in predictions:
        text_predictions.append({'prediction': predict[1], 'confidence': predict[2]})
        
    processed_text_predictions = []
    for predict in text_predictions:
        prediction = predict['prediction']
        confidence = float(predict['confidence'])  # Convert to float
        processed_text_predictions.append({'prediction': prediction, 'confidence': confidence})

    feature_list = feature.tolist()
    pointsX = [ models.PointStruct(
        id=counter,
        vector=feature_list,
        payload={
            "file_name": img_name,
            "url": img_url,
            "tags": processed_text_predictions
            },
        )
    ]
    
    points = {
        "file_name": img_name,
        "product_url": img_url,
        "product_image": img_url,
        "tags": processed_text_predictions,
        'vector': feature_list
    }
    
   
    
    client.upsert(collection_name="vector_db", points=pointsX)
    collection.insert(points)    
    del feature_list
    del feature
    del pointsX
    del points
    # collection.flush()
    # print(mr)

    # Define the feature file path
    # feature_path = Path("./static/cache/") / (img_name + ".npy")

    # Save the extracted features to the feature file
    # print('TES222T!')
    # np.save(feature_path, feature)
    



    # Clean up
    del image
    del image_data
    del feature
    del response

def main():
    
    counter = 160543

    # Initialize the FeatureExtractor
    feature_extractor = FeatureExtractor(gpu_mode=True)

    # Load image URLs from images.json file
    with open("images.json", "r") as f:
        data = json.load(f)
        image_urls = data["urls"]

    print(len(image_urls))
    start_time = time.time()
    image_urls = image_urls[160543:]
    # print(len(new_list))
    # print(len(image_urls))
    # print(new_list[0])
    # print(image_urls[0])
    # exit()
    # Process all the images in parallel using multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the image processing tasks to the executor
        futures = []
        for img_url in image_urls:
            img_name = img_url.split("/")[-1].split(".")[0]
            counter += 1
            future = executor.submit(process_image, img_url, feature_extractor, img_name, counter)
            futures.append(future)

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()
            
        end_time = time.time()

        # Calculate the execution time in minutes
        execution_time_minutes = (end_time - start_time) / 60

        print(f"Execution time: {execution_time_minutes:.2f} minutes")

# Example usage
if __name__ == '__main__':
    main()