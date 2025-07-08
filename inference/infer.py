import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import json
import torch.nn.functional as F
from glob import glob

from models.segmentation_module import SegmentationModel
from models.unet import get_unet
from models.losses import get_loss_function

# 🆕 use shared preprocessing
from data.data_loading import prepare_dataset
from data.transformations import get_inference_transforms

def run_inference(input_image_path, output_image_path, model, spatial_size):
    print(f"Running inference on image: {input_image_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Use shared transforms and dataset logic
    df = pd.DataFrame([{"image": input_image_path}])
    transforms = get_inference_transforms(image_size=spatial_size)
    dataset = prepare_dataset(df, transforms)
    sample = dataset[0]

    input_tensor = sample["image"].unsqueeze(0).to(device)
    original_affine = sample["image_affine"]
    original_shape = tuple(sample["image_shape"])

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (C, H, W, D)

    # Resize back to original
    up = F.interpolate(torch.tensor(probs).unsqueeze(0), size=original_shape, mode="trilinear", align_corners=False)
    up = up.squeeze(0).numpy()

    seg = np.argmax(up, axis=0).astype(np.uint8)

    # Save result
    result_nifti = nib.Nifti1Image(seg, affine=original_affine)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    nib.save(result_nifti, output_image_path)
    print(f"Saved predicted segmentation to {output_image_path}")

def main():
    checkpoint_path = "./lightning_logs/best_version/checkpoints/best-checkpoint.ckpt"
    csv_path = "./data/data.csv"
    spatial_size = (128, 128, 128)

    print("Initializing model...")
    model = SegmentationModel.load_from_checkpoint(
        checkpoint_path,
        model=get_unet(),
        loss_function=get_loss_function(name="focal"),
        strict=False
    )

    print("Loading inference data...")
    df = pd.read_csv(csv_path)

    print("Starting inference...")
    for index, row in df.iterrows():
        input_image_path = row["image"]

        # Determine output path
        output_image_path = row.get("outpath", None)
        if not output_image_path or not isinstance(output_image_path, str) or output_image_path.strip() == "":
            # Fallback to adding _BreastSeg before .nii.gz
            base, ext = os.path.splitext(input_image_path)
            if ext == ".gz":  # handle .nii.gz
                base, _ = os.path.splitext(base)
                ext = ".nii.gz"
            output_image_path = base + "_BreastSegI" + ext

        print(f"[{index+1}/{len(df)}] Processing: {input_image_path} -> {output_image_path}")
        run_inference(input_image_path, output_image_path, model, spatial_size)

    print("Inference complete.")

if __name__ == "__main__":
    main()
