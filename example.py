# ============================================================
# STEP 1:
# LOAD THE TRAINED AI MODEL
# ============================================================

# Import your inference utilities
from infer import (
    run_inference,
    SegmentationModel,
    get_unet,
    get_loss_function
)

# ------------------------------------------------------------
# ENTER THE PATH TO YOUR TRAINED MODEL CHECKPOINT HERE
# Example:
# checkpoint_path = r"enter the path here"
# ------------------------------------------------------------

checkpoint_path = r"enter the path here"


# ------------------------------------------------------------
# Load the trained segmentation model
# ------------------------------------------------------------

model = SegmentationModel.load_from_checkpoint(
    checkpoint_path,
    model=get_unet(),
    loss_function=get_loss_function(name="focal"),
    strict=False
)

print("Model loaded successfully.")

# ============================================================
# STEP 2:
# UPLOAD MRI IMAGE -> GENERATE MASK USING AI MODEL
# ============================================================

# ------------------------------------------------------------
# ENTER MRI IMAGE PATH HERE
# This is the MRI scan you want to segment
# ------------------------------------------------------------

input_image_path = r"enter the path here"


# ------------------------------------------------------------
# ENTER OUTPUT MASK PATH HERE
# This is where the generated mask will be saved
# ------------------------------------------------------------

output_mask_path = r"enter the path here"


# ------------------------------------------------------------
# SAME INPUT SIZE USED DURING TRAINING
# ------------------------------------------------------------

spatial_size = (128, 128, 80)


# ------------------------------------------------------------
# RUN AI SEGMENTATION INFERENCE
# This generates the segmentation mask
# ------------------------------------------------------------

run_inference(
    input_image_path=input_image_path,
    output_image_path=output_mask_path,
    model=model,
    spatial_size=spatial_size,
    use_crf=False
)

print("Mask generated successfully.")

# ============================================================
# STEP 3:
# APPLY N4 BIAS FIELD CORRECTION
# USING:
#   1. ORIGINAL MRI IMAGE
#   2. GENERATED MASK
#
# OUTPUT:
#   -> BIAS CORRECTED MRI IMAGE
# ============================================================

import os
import SimpleITK as sitk
import numpy as np


def n4_bias_correct(
    image_path,
    mask_path,
    output_bias_corrected_path,
    output_bias_field_path=None
):
    """
    Applies N4 Bias Field Correction to an MRI image
    using a segmentation mask.

    PARAMETERS
    ----------
    image_path : str
        Path to original MRI scan

    mask_path : str
        Path to generated segmentation mask

    output_bias_corrected_path : str
        Where the corrected MRI will be saved

    output_bias_field_path : str or None
        Optional path to save the estimated bias field
    """

    # --------------------------------------------------------
    # READ ORIGINAL MRI IMAGE
    # --------------------------------------------------------

    image = sitk.ReadImage(image_path)

    # Save original datatype
    original_dtype = image.GetPixelID()

    # Convert image to float32
    image_float = sitk.Cast(image, sitk.sitkFloat32)


    # --------------------------------------------------------
    # READ SEGMENTATION MASK
    # --------------------------------------------------------

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found:\n{mask_path}")

    mask = sitk.ReadImage(mask_path)

    # Convert mask into binary format
    # Everything > 0 becomes 1
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)


    # --------------------------------------------------------
    # CREATE N4 BIAS FIELD CORRECTION FILTER
    # --------------------------------------------------------

    n4_filter = sitk.N4BiasFieldCorrectionImageFilter()

    # Number of iterations at each resolution level
    n4_filter.SetMaximumNumberOfIterations((50, 50, 50, 50))


    # --------------------------------------------------------
    # APPLY BIAS FIELD CORRECTION
    # --------------------------------------------------------

    corrected_image = n4_filter.Execute(
        image_float,
        mask
    )


    # --------------------------------------------------------
    # CONVERT BACK TO ORIGINAL DATATYPE
    # --------------------------------------------------------

    corrected_image = sitk.Cast(
        corrected_image,
        original_dtype
    )


    # --------------------------------------------------------
    # SAVE BIAS CORRECTED MRI
    # --------------------------------------------------------

    sitk.WriteImage(
        corrected_image,
        output_bias_corrected_path
    )

    print("\nBias corrected MRI saved:")
    print(output_bias_corrected_path)


    # --------------------------------------------------------
    # OPTIONAL:
    # SAVE THE ESTIMATED BIAS FIELD
    # --------------------------------------------------------

    if output_bias_field_path is not None:

        log_bias_field = n4_filter.GetLogBiasFieldAsImage(
            image_float
        )

        sitk.WriteImage(
            log_bias_field,
            output_bias_field_path
        )

        print("\nBias field saved:")
        print(output_bias_field_path)


    return corrected_image


# ============================================================
# STEP 4:
# RUN BIAS CORRECTION
# ============================================================

# ------------------------------------------------------------
# ORIGINAL MRI IMAGE
# ------------------------------------------------------------

image_path = r"enter the path here"


# ------------------------------------------------------------
# GENERATED AI MASK
# ------------------------------------------------------------

mask_path = r"enter the path here"


# ------------------------------------------------------------
# OUTPUT BIAS CORRECTED MRI
# ------------------------------------------------------------

output_bias_corrected_path = r"enter the path here"


# ------------------------------------------------------------
# OPTIONAL:
# SAVE THE BIAS FIELD IMAGE
# ------------------------------------------------------------

output_bias_field_path = r"enter the path here"


# ------------------------------------------------------------
# RUN N4 BIAS FIELD CORRECTION
# ------------------------------------------------------------

corrected_image = n4_bias_correct(
    image_path=image_path,
    mask_path=mask_path,
    output_bias_corrected_path=output_bias_corrected_path,
    output_bias_field_path=output_bias_field_path
)

print("\nBias correction completed successfully.")

# ============================================================
# STEP 5:
# NOW USE THE BIAS CORRECTED MRI AGAIN
# TO GENERATE A NEW MASK
#
# INPUT:
#   -> BIAS CORRECTED MRI
#
# OUTPUT:
#   -> NEW MASK FOR BIAS CORRECTED MRI
# ============================================================

# ------------------------------------------------------------
# ENTER BIAS CORRECTED MRI PATH
# ------------------------------------------------------------

bias_corrected_mri_path = r"enter the path here"


# ------------------------------------------------------------
# ENTER OUTPUT PATH FOR NEW MASK
# ------------------------------------------------------------

bias_corrected_mask_output_path = r"enter the path here"


# ------------------------------------------------------------
# RUN INFERENCE AGAIN
# ------------------------------------------------------------

run_inference(
    input_image_path=bias_corrected_mri_path,
    output_image_path=bias_corrected_mask_output_path,
    model=model,
    spatial_size=(128, 128, 80),
    use_crf=False
)

print("\nNew mask for bias corrected MRI generated successfully.")

