import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import torch
from monai.visualize import GradCAM
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')


class GradCAMCallback(Callback):
    def __init__(self, model, val_loader, target_layers, class_idx=1, log_samples=4):
        self.model = model.module if isinstance(model, torch.nn.DataParallel) else model
        self.val_loader = val_loader
        self.target_layers = target_layers
        self.class_idx = class_idx
        self.log_samples = log_samples

    def log_gradcam_with_temporal_slider(self, trainer, inputs, cam, tag, epoch):
        if trainer.global_rank != 0:
            return  # Log only on primary process
        central_slice_idx = inputs.shape[-1] // 2
        for sample_idx in range(min(inputs.shape[0], self.log_samples)):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            image_np = inputs[sample_idx, 0, :, :, central_slice_idx].detach().cpu().numpy()
            cam_np = cam[sample_idx, 0, :, :, central_slice_idx].detach().cpu().numpy()
            ax.imshow(image_np, cmap="gray")
            ax.imshow(cam_np, cmap="jet", alpha=0.5)
            ax.axis("off")
            trainer.logger.experiment.add_figure(f"{tag}/sample_{sample_idx}", fig, global_step=epoch)
            plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module):
        gradcam = GradCAM(nn_module=self.model, target_layers=self.target_layers)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= self.log_samples:
                    break
                inputs = batch["image"].to(pl_module.device)
                inputs.requires_grad_()
                with torch.enable_grad():
                    cam = gradcam(x=inputs, class_idx=self.class_idx)
                self.log_gradcam_with_temporal_slider(trainer, inputs, cam, tag="GradCAM_Temporal", epoch=trainer.current_epoch)


def train_model(train_loader, val_loader, model, loss_function, max_epochs=200, resume=None):
    from models.segmentation_module import SegmentationModel
    segmentation_model = SegmentationModel(model=model, loss_function=loss_function)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-checkpoint", verbose=True)
    early_stopping = EarlyStopping(monitor="val_loss", patience=15, mode="min", verbose=False)
    gradcam_callback = GradCAMCallback(model=model, val_loader=val_loader, target_layers=["model.2.1.conv.unit0.conv"], class_idx=1)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping, gradcam_callback],
        accelerator="gpu",
        devices=1,#torch.cuda.device_count(),
        strategy='auto',#"ddp",
        log_every_n_steps=10,
    )
    trainer.fit(segmentation_model, train_loader, val_loader, ckpt_path=resume)
