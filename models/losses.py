from monai.losses import DiceCELoss, FocalLoss

def get_loss_function(name="focal"):
    """
    Returns the requested loss function.
    """
    if name == "focal":
        return FocalLoss(to_onehot_y=True, gamma=2.0)
    elif name == "dicece":
        return DiceCELoss(to_onehot_y=True, softmax=True)
    else:
        raise ValueError(f"Unknown loss function: {name}")
