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


import subprocess


def get_cuda_version():
    try:
        try:
            import torch
            cuda_version = torch.version.cuda
            return cuda_version
        except:
            nvcc_output = subprocess.check_output("nvcc --version", shell=True).decode()
            cuda_version = re.search(r"release (\d+\.\d+)", nvcc_output).group(1)
            return cuda_version

    except Exception as e:
        print(f"Error getting CUDA version: {e}")
        return None

import sys
def install_missing_packages():
    # Extract the major, minor, and micro parts of the Python version
    major, minor, micro = sys.version_info[:3]

    # Define the packages list based on the Python version
    if (major, minor, micro) >= (3, 8, 10) and (major, minor, micro) <= (3, 9, 18):
        packages = [
            "numpy==1.23.4",
            "sigpy==0.1.25",
            "matplotlib==3.7.2",
            "opencv-python"
        ]
    elif (major, minor, micro) > (3, 9, 18):
        packages = [
            "numpy",
            "sigpy",
            "matplotlib",
            "opencv-python"
        ]
    else:
        #print in red
        print("\033[91m" + "Your Python version is quite low and may encounter compatibility issues!" + "\033[0m")
        packages = [
            "numpy==1.23.4",
            "sigpy==0.1.25",
            "matplotlib==3.7.2",
            "opencv-python"
        ]

    cuda_version = get_cuda_version()

    # Add CuPy package based on CUDA version
    if cuda_version:
        if cuda_version == '11.8':
            packages.append(f"cupy==11.3.0")
        else:
            packages.append(f"cupy-cuda{cuda_version.replace('.', '')}")
    else:
        print("CUDA version not found. Skipping CuPy installation.")

    for package in packages:
        print(f"Installing {package}...")
        subprocess.run(["pip", "install", package])

    # Install NCCL 
    # install_package_with_conda('nccl')

    print("Installation of missing packages complete!")


import requests
import zipfile
import os
import subprocess
import pkg_resources

def download_file(url, target_path):
    response = requests.get(url, stream=True)
    handle = open(target_path, "wb")
    for chunk in response.iter_content(chunk_size=512):
        if chunk:  # filter out keep-alive new chunks
            handle.write(chunk)
    handle.close()

def check_numpy_version():
    required_version = (1, 17, 3)
    
    try:
        numpy_version = pkg_resources.get_distribution("numpy").version
        installed_version = tuple(map(int, numpy_version.split('.')))

        if installed_version < required_version:
            return False
        else:
            return True

    except pkg_resources.DistributionNotFound:
        return False
    
def copy_file(src, dest):
    """Manually copy the contents of one file to another."""
    with open(src, 'r') as source:
        content = source.read()
    with open(dest, 'w') as destination:
        destination.write(content)

def recursive_delete_dir(target_dir):
    # List all files and directories in the target directory
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        
        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Recursively delete the sub-directory
            recursive_delete_dir(item_path)
        else:
            # Remove the file
            os.remove(item_path)
    
    # Once the directory is empty, delete it
    os.rmdir(target_dir)

def install_twixtools(auto_install_missing_packages = False):

    # Downloading the twixtools pip file
    print("Downloading twixtools...")
    
    # download the *sdist* (tar.gz) instead of the wheel
    subprocess.check_call([
        sys.executable, "-m", "pip", "download", "twixtools",
        "--no-binary", ":all:",      # <-- crucial
        "--no-deps", "-d", "."
    ])
    
    src = glob.glob("twixtools-*.tar.gz")[0]        # e.g. twixtools-0.23.tar.gz
    src_name = src.split('.tar.gz')[0]
    print("twixtools package downloaded, version: ", src_name)
    
    # unpack into twixtools-*/
    with tarfile.open(src) as tf:
        tf.extractall(".")

    # Replace twix_map.py with the one in current directory
    print("Replacing twix_map.py and geometry.py...")
    try:
        copy_file('./helper_functions/map_twix.py', './'+src_name+'/twixtools/map_twix.py')
        copy_file('./helper_functions/geometry.py', './'+src_name+'/twixtools/geometry.py')
    except FileNotFoundError:
        #download the modified twixtools
        url_map_twix = "https://figshare.com/ndownloader/files/41951475"
        download_file_from_figshare('./helper_functions/',url_map_twix)
        copy_file('./helper_functions/map_twix.py', './'+src_name+'/twixtools/map_twix.py')
        os.remove('./helper_functions/map_twix.py')
        #download the modified geometry.py
        url_geomery_twix = "https://figshare.com/ndownloader/files/55351121"
        download_file_from_figshare('./helper_functions/',url_geomery_twix)
        copy_file('./helper_functions/geometry.py', './'+src_name+'/twixtools/geometry.py')
        os.remove('./helper_functions/geometry.py')
    

    print("Installing twixtools...")
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", src_name],
        capture_output=True,   # collects stdout & stderr
        text=True
    )

    if result.returncode != 0:
        print("An error occurred while installing twixtools. Please contact the author with the following information.")
        print("STDERR:\n", result.stderr)

    # Removing the zip file and the extracted directory
    print("Removing the zip file and the extracted directory...")
    os.remove(src)
    recursive_delete_dir(src_name)

    if auto_install_missing_packages:
        # Check and install numpy if required
        if not check_numpy_version():
            print("Installing numpy...")
            subprocess.run(["pip", "install", "numpy>=1.17.3"])
        else:
            print("Sufficient numpy version is already installed.")
        #install other missing packages
        install_missing_packages()
    else:
        print("Ensure you have installed the following packages manually:")
        print("For Python version between 3.8.10 ~ 3.9.18")
        print("numpy==1.23.4, sigpy==0.1.25, matplotlib==3.7.2, opencv-python")
        print("For Python version higher than 3.9.18")
        print("numpy, sigpy, matplotlib, opencv-python")
    
    print("Installation complete!")
