import csv
import re
import json
import urllib.request
import os
from concurrent.futures import ThreadPoolExecutor


def extract_urls_from_csv(file_path):
    urls = []
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        
        for row in reader:
            # Assuming URLs are in a specific column, adjust the column index accordingly
            # text = row['url']
            text = row[0]
            # text = row[2]
            
            # Use regular expression to find URLs in the text
            url_matches = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
            
            urls.extend(url_matches)
    
    return urls

def write_urls_to_json(urls, file_path):
    data = {'urls': urls}
    
    with open(file_path, 'w') as file:
        json.dump(data, file)
        
        
def download_image(url, output_folder):
    try:
        filename = os.path.basename(url)
        image_path = os.path.join(output_folder, filename)
        urllib.request.urlretrieve(url, image_path)
        print(f'Successfully downloaded {url} to {image_path}')
    except Exception as e:
        print(f'Error downloading {url}: {e}')

def download_images_from_json(json_file_path, output_folder):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        urls = data['urls']

        with ThreadPoolExecutor() as executor:
            futures = []
            for url in urls:
                future = executor.submit(download_image, url, output_folder)
                futures.append(future)

            # Wait for all tasks to complete
            for future in futures:
                future.result()

def jsonlTojson():
    # Open the JSONL file for reading
    with open('input.jsonl', 'r') as file:
        urls = []

        # Read each line (JSON object) in the file
        for line in file:
            data = json.loads(line)

            # Extract the "originalSrc" value from the "image" object
            url = data['image']['originalSrc']
            urls.append(url)

    # Create a dictionary with the URLs
    output = {'urls': urls}

    # Write the dictionary to the "images.json" file
    with open('images.json', 'w') as file:
        json.dump(output, file)

urls = extract_urls_from_csv('dorsiabc__lookalike.csv')
write_urls_to_json(urls, 'images.json')
# download_images_from_json('images.json', 'static/dataset')