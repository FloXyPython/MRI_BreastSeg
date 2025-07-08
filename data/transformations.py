import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityd,
    Resized, RandFlipd, RandRotate90d, RandShiftIntensityd, Rand3DElasticd,
    RandZoomd, RandAffined, RandGaussianNoised, RandAdjustContrastd, MapTransform
)
import nibabel as nib

class AddMetadataFromNifti(MapTransform):
    """
    Custom transform to extract affine and shape metadata from NIfTI images.
    """
    def __init__(self, keys):
        super().__init__(keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            nifti = nib.load(d[key])
            d[key + "_affine"] = nifti.affine
            d[key + "_shape"] = np.array(nifti.shape)
        return d

class RandAxisSwapd(MapTransform):
    """
    Randomly swaps axes of the input 3D array.
    """
    def __init__(self, keys, prob=0.5):
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        if np.random.rand() < self.prob:
            # Generate a random permutation of axes
            perm = np.random.permutation(3)
            for key in self.keys:
                d[key] = np.transpose(d[key], axes=perm)
        return d

def get_train_transforms(image_size=(128, 128, 80)):
    """
    Get data augmentation and preprocessing transforms for training.
    
    Args:
        image_size (tuple): Desired spatial size for resizing images and labels.
    
    Returns:
        Compose: MONAI transform pipeline.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),  # Ensure channel-first format
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        Resized(keys=["image", "label"], spatial_size=image_size),  # Resize to uniform size
        #RandAxisSwapd(keys=["image", "label"], prob=0.5),  # Random axis swapping
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),  # Random axis flipping
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
        Rand3DElasticd(
            keys=["image", "label"], sigma_range=(5, 8), magnitude_range=(100, 200), 
            prob=0.3, mode=("bilinear", "nearest")
        ),
        RandZoomd(keys=["image", "label"], min_zoom=0.5, max_zoom=1.5, prob=0.3, mode=("trilinear", "nearest")),  # Adjusted zoom
        RandAffined(
            keys=["image", "label"], rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1),
            shear_range=(0.05, 0.05, 0.05), mode=("bilinear", "nearest"), prob=0.3
        ),
        RandGaussianNoised(keys=["image"], mean=0.0, std=0.05, prob=0.2),
        RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 2.0)),
    ])

def get_val_transforms(image_size=(128, 128, 80)):
    """
    Get preprocessing transforms for validation and testing (no augmentations).
    
    Args:
        image_size (tuple): Desired spatial size for resizing images and labels.
    
    Returns:
        Compose: MONAI transform pipeline.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        Resized(keys=["image", "label"], spatial_size=image_size),
    ])

def get_test_transforms(image_size=(128, 128, 80)):
    """
    Get preprocessing transforms for testing (same as validation).
    
    Args:
        image_size (tuple): Desired spatial size for resizing images and labels.
    
    Returns:
        Compose: MONAI transform pipeline.
    """
    return get_val_transforms(image_size=image_size)


def get_inference_transforms(image_size=(128, 128, 80)):
    """
    Get transforms for inference — no label, includes metadata extraction.
    """
    return Compose([
        AddMetadataFromNifti(keys=["image"]),
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        Resized(keys=["image"], spatial_size=image_size),
    ])

