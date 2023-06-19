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
            text = row[2]
            
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


urls = extract_urls_from_csv('dorsiabc_dev__lookalike-230619.csv')
write_urls_to_json(urls, 'images.json')
download_images_from_json('images.json', 'static/dataset')