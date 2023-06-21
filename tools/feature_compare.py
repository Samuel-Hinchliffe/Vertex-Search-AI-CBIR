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

def compare_vectors(folder_path1, folder_path2, output_file):
    npy_files1 = get_npy_files(folder_path1)
    npy_files2 = get_npy_files(folder_path2)

    with open(output_file, 'w') as f:
        for file1 in npy_files1:
            filename = os.path.basename(file1)
            file2 = os.path.join(folder_path2, filename)

            if file2 in npy_files2:
                try:
                    vector1 = np.load(file1)
                    vector2 = np.load(file2)
                    distance = np.linalg.norm(vector1 - vector2)
                    f.write(f"Filename: {filename}, Distance: {distance}\n")
                except ValueError:
                    f.write(f"Error reading file: {filename}\n")
            else:
                f.write(f"Matching file not found for: {filename}\n")

def get_npy_files(folder_path):
    file_pattern = folder_path + "/*.npy"
    npy_files = glob.glob(file_pattern)
    return npy_files


output_file = "comparison_output2.txt"  # Replace with the desired output file name or path
compare_vectors('./static/cache', './static/cache - Internet', output_file)
