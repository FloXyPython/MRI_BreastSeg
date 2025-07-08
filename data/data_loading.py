import os
import pandas as pd
from sklearn.model_selection import train_test_split
from monai.data import Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized

def load_csv_data(csv_path):
    """
    Load the CSV file containing image and segmentation paths.
    
    Args:
        csv_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame containing the data.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    data = pd.read_csv(csv_path)
    if 'image' not in data.columns or 'label' not in data.columns:
        raise ValueError("CSV file must contain 'image' and 'label' columns.")
    
    return data

def split_data(data, test_size=0.2, random_state=42):
    """
    Split the data into training and validation sets.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        test_size (float): Proportion of the data to use as the test set.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        (pd.DataFrame, pd.DataFrame): Train and test splits as DataFrames.
    """
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

def prepare_dataset(data, transforms):
    """
    Prepare a MONAI Dataset from a DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data with 'image' and 'label' columns.
        transforms (Compose): MONAI transforms to apply to the dataset.
    
    Returns:
        Dataset: MONAI Dataset object.
    """
    # Convert DataFrame to a list of dictionaries
    #data_list = data.to_dict(orient='records')
    # Create MONAI Dataset
    #return Dataset(data=data_list, transform=transforms)
    data_list = []
    for rec in data.to_dict(orient='records'):
        rec['id'] = os.path.basename(rec['image'])
        data_list.append(rec)
    return Dataset(data=data_list, transform=transforms)
    
    

def get_default_transforms(image_size=(128, 128, 128)):
    """
    Get default MONAI transforms for images and labels.
    
    Args:
        image_size (tuple): Target size for resizing images and labels.
    
    Returns:
        Compose: A composed list of MONAI transforms.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Resized(keys=["image", "label"], spatial_size=image_size)
    ])
