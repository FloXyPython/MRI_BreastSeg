from data.data_loading import load_csv_data, split_data, prepare_dataset
from data.transformations import get_train_transforms, get_val_transforms
from training.train import train_model
from models.unet import get_unet, get_attention_unet
from models.losses import get_loss_function
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_lightning as pl
import torch

# Configuration Constants
CSV_FILE = "data/data.csv"        # Path to the CSV file
BATCH_SIZE = 8                   # Batch size for training and testing
NUM_WORKERS = 16                   # Number of workers for DataLoader
IMAGE_SIZE = (128, 128, 128)      # Desired image size
MAX_EPOCHS = 200                  # Maximum number of training epochs
TEST_SPLIT_RATIO = 0.2            # Ratio of data to use for testing
LOSS_FUNCTION_NAME = "focal"      # Choose between "focal" and "dicece"

def main():

    # Step 0: Initialize seed
    pl.seed_everything(42)

    # Step 1: Load Data
    print("Loading data...")
    data = load_csv_data(CSV_FILE)

    # Step 2: Split Data
    print("Splitting data into train and test sets...")
    train_data, test_data = split_data(data, test_size=TEST_SPLIT_RATIO)

    # Step 3: Define Transforms
    print("Preparing transforms...")
    train_transforms = get_train_transforms(image_size=IMAGE_SIZE)
    test_transforms = get_val_transforms(image_size=IMAGE_SIZE)

    # Step 4: Prepare Datasets
    print("Preparing datasets...")
    test_dataset = prepare_dataset(test_data, test_transforms)
    train_dataset = prepare_dataset(train_data, train_transforms)

    # Step 5: Create DataLoaders 
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Step 6: Define Model and Loss Function
    print("Initializing model and loss function...")
    model = get_unet()
    #model = get_attention_unet()
    #print(model)
    #return
    loss_function = get_loss_function(name=LOSS_FUNCTION_NAME)

    # Step 7: Train the Model
    print("Starting training...")
    train_model(train_loader, test_loader, model, loss_function, max_epochs=MAX_EPOCHS)#,resume='./lightning_logs/version_60/checkpoints/best-checkpoint.ckpt')

if __name__ == "__main__":
    main()
