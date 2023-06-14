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

def process_image(img_path, feature_extractor):
    # Print the image file path
    # print(img_path)

    # Open the image using PIL
    image = Image.open(img_path)

    # Extract features from the image using the FeatureExtractor class
    feature = feature_extractor.extract(img=image)

    # Define the feature file path
    feature_path = Path("./static/cache/") / (img_path.stem + ".npy")

    # Save the extracted features to the feature file
    np.save(feature_path, feature)

def main():
    # Initialize the FeatureExtractor
    feature_extractor = FeatureExtractor()

    # Define the image file extensions to search for
    extensions = [".jpg", ".jpeg", ".webp", ".png"]

    # Get the dataset directory (Your images!)
    dataset_dir = Path("./static/dataset")

    # Get the paths of all image files in the dataset directory
    image_paths = []
    for extension in extensions:
        image_paths.extend(dataset_dir.glob(f"*{extension}"))
    
    # Process all the images in parallel using multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the image processing tasks to the executor
        futures = []
        for img_path in image_paths:
            future = executor.submit(process_image, img_path, feature_extractor)
            futures.append(future)

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()

# Example usage
if __name__ == '__main__':
    main()