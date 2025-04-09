import os
import kagglehub

def download_dataset():
    """
    Download or use the local aerial images dataset of roundabouts.
    
    Returns:
        str: The path to the dataset.
    """
    # Define the relative path
    dataset_path = os.path.join("data", "roundabout-aerial-images-for-vehicle-detection")
    # Check if the dataset already exists
    if os.path.exists(dataset_path):
        print(f"The dataset already exists. Skipping the download. Path: {dataset_path}")
        return dataset_path
    
    # If the dataset doesn't exist locally, try to download it from Kaggle
    try:
        print("Trying to download the dataset from Kaggle...")
        # Use kagglehub to download the dataset according to the official example code
        path = kagglehub.dataset_download("javiersanchezsoriano/roundabout-aerial-images-for-vehicle-detection")
        print(f"The dataset has been downloaded. Path: {path}")
        return path
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")
        print("Please make sure kagglehub is installed: pip install kagglehub")
    
    # If the download fails, try to use the existing local path
    local_path = "D:/学习软件/CDS521/vehicle detection/data/kagglehub/datasets/javiersanchezsoriano/roundabout-aerial-images-for-vehicle-detection/versions/2"
    if os.path.exists(local_path):
        print(f"Using the existing local dataset path: {local_path}")
        return local_path
    
    print("Warning: Unable to download the dataset and the local path does not exist.")
    print("Please manually download the dataset or ensure the local path is correct.")
    return None

if __name__ == "__main__":
    dataset_path = download_dataset()
    print(f"\nDataset path: {dataset_path}")