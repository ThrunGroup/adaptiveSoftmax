import gdown
import os
from mnl.mnl_constants import (
    GOOGLE_DRIVE_PREFIX,
    MNL_WEIGHTS_DIR,
    MNL_XS_DIR,
    MNIST_FINAL_PATH,
    EUROSAT_FINAL_PATH,
)

def download(files_info, dest):
    os.makedirs(dest, exist_ok=True)
    for link, name in files_info: 
        output_path = os.path.join(dest, name)  
        gdown.download(link, output_path, quiet=False)

def download_weights():
    weight_files = [
        (f'{GOOGLE_DRIVE_PREFIX}18WX7pBFedfD6Yxu3bIEV6WbhWWVSL7j7', MNIST_FINAL_PATH),
        (f'{GOOGLE_DRIVE_PREFIX}18MuJ-wrI7CtkGGzlcWSc-f8yOX6qIHvY', EUROSAT_FINAL_PATH),
    ]
    download(weight_files, MNL_WEIGHTS_DIR)

def download_queries():
    weight_files = [
        (f'{GOOGLE_DRIVE_PREFIX}1791SCBmV9dUu-v6wYteSAWNA8Vynkzpf', MNIST_FINAL_PATH),
        (f'{GOOGLE_DRIVE_PREFIX}176VZHHvTPyY6NoxgzH6S_ZjaZRX6q-3x', EUROSAT_FINAL_PATH),
    ]
    download(weight_files, MNL_XS_DIR)

if __name__ == "__main__":
    download_weights()
    download_queries()


