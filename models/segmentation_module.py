import torch
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from torch.optim import Adam
from matplotlib import pyplot as plt
from monai.visualize import GradCAM


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, loss_function, learning_rate=1e-4, num_classes=2):
        super(SegmentationModel, self).__init__()
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.num_classes = num_classes

    @staticmethod
    def for_inference(model, num_classes=2):
        """
        Create a SegmentationModel instance specifically for inference.
        """
        return SegmentationModel(model=model, loss_function=None, num_classes=num_classes)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, model, loss_function, *args, **kwargs):
        """
        Custom load_from_checkpoint to pass additional arguments required by __init__.
        """
        instance = super(SegmentationModel, cls).load_from_checkpoint(
            checkpoint_path, model=model, loss_function=loss_function, *args, **kwargs
        )
        return instance

    def forward(self, x):
        return self.model(x)


    def step(self, batch, batch_idx, stage="train"):
        inputs, labels = batch["image"], batch["label"]

        # Forward pass
        outputs = self.model(inputs)

        # Calculate loss
        loss = self.loss_function(outputs, labels) if self.loss_function else None

        # Convert outputs to predicted classes using argmax
        predicted_classes = torch.argmax(outputs, dim=1)

        # One-hot encode predicted classes and ground truth
        predicted_classes_one_hot = one_hot(predicted_classes.unsqueeze(1), num_classes=self.num_classes)
        labels_one_hot = one_hot(labels, num_classes=self.num_classes)

        # Compute Dice score
        self.dice_metric(y_pred=predicted_classes_one_hot, y=labels_one_hot)
        dice_score = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        # Log metrics
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, batch_size=inputs.size(0),sync_dist=True)
        self.log(f"{stage}_dice", dice_score, on_step=False, on_epoch=True, prog_bar=True, batch_size=inputs.size(0),sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="val")

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
