import requests
import time
import re

def download_file_from_figshare(saving_path, url):
    # Step 1: Access the Figshare link to get the direct download link.
    response = requests.get(url, allow_redirects=False)  # Disallow redirects so we can capture the redirection URL.
    if response.status_code != 302 or 'Location' not in response.headers:
        raise ValueError("Unable to obtain direct download link.")
    
    download_link = response.headers['Location']

    # Step 2: Download the content from the direct link.
    response = requests.get(download_link)
    response.raise_for_status()  # Raise an error if the request fails.
    
    #get the time stamp
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Extract the filename from the URL or use a default name.
    match = re.search(r"/([^/]+)\?", download_link)
    filename = match.group(1) if match else 'dataset_' + url.split('/')[-1].split('?')[0] + '.dat' or "dataset_"+ timestr +".dat"

    # Step 3: Save the content to a local file.
    with open(saving_path+filename, 'wb') as file:
        file.write(response.content)
    #print with green color
    print("\033[92m" + f"File saved as {filename}" + "\033[0m")
