import torch
from monai.visualize import GradCAM

def apply_gradcam(model, input_image, target_layer, label_index):
    """
    Apply Grad-CAM on a 3D model and return the activation map.
    """
    cam = GradCAM(nn_module=model, target_layers=[target_layer])
    cam_map = cam(x=input_image, class_idx=label_index)
    return cam_map.squeeze(0).cpu().numpy()  # Remove batch dimension
