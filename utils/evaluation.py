# utils/evaluation.py

import numpy as np
def dice_score(pred, target, smooth=1e-8):
    pred = (pred>0).astype(np.float32)
    target = (target > 0).astype(np.float32)

    intersection = np.sum(pred * target)
    dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
    return dice