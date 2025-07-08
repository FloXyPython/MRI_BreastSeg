from monai.networks.nets import UNet, AttentionUnet, DynUNet
import torch

def get_unet():
    """
    Returns a 3D U-Net model.
    """
    model = UNet(spatial_dims=3,
                 in_channels=1,
                 out_channels=2,
                 channels=(32, 64, 128, 256, 512),
                 strides=(2, 2, 2, 2),
                 num_res_units=4,
                 dropout=0.2,
                 kernel_size=3,
                 up_kernel_size=3,
                 act="leakyrelu",
                 norm="instance",
                 bias=True
    )
    return model.cuda()

def get_attention_unet():
    """
    Returns a 3D Attention U-Net model with optimized parameters.
    """
    model =  AttentionUnet(spatial_dims=3,                     # 3D for volumetric data
                           in_channels=1,                      # Input channels (e.g., grayscale MRI)
                           out_channels=2,                     # Output channels (background + breast mask)
                           channels=(32, 64, 128, 256, 512),   # Encoder/decoder feature maps
                           strides=(2, 2, 2, 2),               # Downsampling strides
                           dropout=0.3,                        # Increased dropout for regularization
                           kernel_size=3,                      # Default kernel size for convolutional layers
                           up_kernel_size=3,                   # Kernel size for upsampling layers
    )
    return model.cuda()

def get_dynunet():
    """
    Returns a 3D DynUNet model.
    """
    return DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        kernel_size=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        filters=[32, 64, 128, 256, 320],
    )
