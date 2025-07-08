# MRI_BreastSeg

This repository contains a deep learning model for the segmentation of breast tissue in MRI scans. The model is trained using the [**MAMA-MIA**](https://github.com/LidiaGarrucho/MAMA-MIA) dataset and some additional private data. It uses a custom **U-Net**, a convolutional neural network architecture commonly used for segmentation tasks.

## Table of Contents

- [Dataset](#dataset)
- [Model](#model)
- [Augmentation Strategy](#augmentation-strategy)
- [Training Procedure](#training-procedure)
- [Results](#results)
- [How to Use](#how-to-use)
- [References](#references)

## Dataset

The model is trained and evaluated using the [**MAMA-MIA**](https://github.com/LidiaGarrucho/MAMA-MIA) dataset, which is a collection of 4 different datasets: DUKE, ISPY1, ISPY2 and NACT and contains breast MRI scans of a total of 1506 subjects, as well as some private data. Ground truth for breast tissue segmentation was obtained by manual segmentation on a subset of each sub-dataset by an expert grader, except for the **DUKE** dataset. For the DUKE dataset we used the publicly available [**3D-Breast-FGT-and-Blood-Vessel-Segmentation**](https://github.com/mazurowski-lab/3D-Breast-FGT-and-Blood-Vessel-Segmentation) segmentation model, as it was originally trained on the DUKE dataset and performed well on that data (not so much on the other datasets though).
|Dataset|Total number of Subjects|Number of Segmentation Masks|Train|Validation|
|---|---|---|---|---|
|DUKE|291|291|281|10|
|ISPY1|171|33|23|10|
|ISPY2|980|37|27|10|
|NACT|64|24|14|10|
|Private Set A|10|10|0|10|
|Private Set B|45|45|35|10|
|<b>total</b>|<b>1561</b>|<b>440</b>|<b>380</b>|<b>60</b>|


## Model

We used the **U-Net** architecture for segmentation, implemented using the MONAI library. U-Net is particularly well-suited for medical image segmentation tasks due to its encoder-decoder structure, which helps preserve fine details in segmentation masks.

**Model architecture**: 
- **U-Net** (monai)
- Input: 3D MRI images
- Output: Binary segmentation masks (breast tissue vs background)

## Augmentation Strategy

Given the high variability in the orientation of breast MRI data, our augmentation pipeline is designed to handle different views and orientations, ensuring that the model can generalize to any input orientation.

### Applied Augmentations:
- **Random flipping**: Handles changes in orientation across the x, y, and z axes.
- **Random rotations**: Covers a wide range of orientations, including those often seen in breast MRI scans.
- **Elastic deformations**: Simulates local deformations of breast tissue to increase robustness.
- **Random intensity and contrast adjustments**: To ensure the model handles variations in intensity that might be present in different scans.
- **Random zooming**: Helps improve the model’s ability to handle images with varying scale and resolution.

## Training Procedure
The model was trained on a NVIDIA A40 GPU (virtualized) with 12GB VRAM on Ubuntu. Training parameters were chosen to fit into hardware constraints.

Training is conducted using **PyTorch Lightning** for better management of the training loop. Key aspects of the training process include:

- **Loss function**: We used the **Focal Loss** to help the model focus on difficult-to-segment regions, especially in class-imbalanced data.
- **Early stopping**: Training halts early if the validation loss does not improve for 15 consecutive epochs.
- **GradCAM**: The **GradCAM** callback is used to visualize class activation maps, providing insights into what areas of the image the model is focusing on.

### Training Configuration:
- **Batch size**: 8
- **Epochs**: 200 (maximum)
- **GPU acceleration**: Training is performed using a GPU for faster processing.
- **Learning rate**: Default values, with potential for adjustments.
  
The model is trained using the following command:

```bash
python main.py
```

The training script utilizes callbacks for model checkpointing, early stopping, and GradCAM visualization.

## Results

Model performance was evaluated using **DICE coefficient** (DICE score) on the initially set-aside validation set (see [Dataset](#dataset)). The DICE score is calculated as:

$$\text{DICE} = \frac{2 \times |A \cap B|}{|A| + |B|}$$

Where **A** is the set of predicted pixels, and **B** is the set of ground truth pixels.

**DICE Evaluation Results** will be presented in the following table:

|Dataset|Dice Score|
|---|---|
|DUKE|0.881|
|ISPY1|0.877|
|ISPY2|0.880|
|NACT|0.891|
|Private Set A|0.859|
|Private Set B|0.898|
|<b>Overall</b>|<b>0.881</b>|

## How to Use
Create a new virtual environment (Python=3.9) and install the requirements as stated in [requirements.txt](./requirements.txt) via
```bash
pip install -r requirements.txt
```

### Use the model for inference
1. **Prepare the dataset**: Ensure the MRI images are in NIFTI format. 
2. **Prepare the csv**: Create an inference csv with the columns

  |subject|image|outpath|
  |---|---|---|
  |Subject_01|<path_to_NIFTI_image>|<filename_of_output>|
  |Subject_02|<path_to_NIFTI_image>|<filename_of_output>|
  |...|...|...|

  With the outpath column being optional. if not given or set to `None` then the Segmentation will be saved in the same folder as the incoming image.

3. **Adjust the path**: Update the `csv_path` in [infer.py](./inference/infer.py) to point to your new csv.
4. **Run inference**: call inference using 
```bash
python -m inference.infer
```
### Train the model 
1. **Prepare the dataset**: Ensure the MRI images and the segmentation masks are in NIFTI format (uint8 for masks, 0=Background, 1=Foreground)
2. **Prepare the csv**: Create a training csv with the columns

  |subject|image|label|
  |---|---|---|
  |Subject_01|<path_to_NIFTI_image>|<path_to_segmentation>|
  |Subject_02|<path_to_NIFTI_image>|<path_to_segmentation>|
  |...|...|...|

3. **Adjust the path**: Update the `CSV_FILE` in the [main function](./main.py) to point to your new csv.
4. **(Optional) Update configuration**: update things like batch size and number of epochs to fit your requirements
5. **Start training**: run training using
```bash
python main.py
```


## References

- **MAMA-MIA dataset**: [https://github.com/LidiaGarrucho/MAMA-MIA]
- **3D-Breast-FGT-and-Blood-Vessel-Segmentation**: [https://github.com/mazurowski-lab/3D-Breast-FGT-and-Blood-Vessel-Segmentation]
