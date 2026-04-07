import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader
import json

class MetContrastiveModel(pl.LightningModule):
    def __init__(self, d_image=384, d_text=768, d_joint=512, learning_rate=1e-4, temperature_init=0.07):
        super().__init__()
        self.save_hyperparameters()
        
        self.W_image = nn.Linear(d_image, d_joint)
        self.W_text = nn.Linear(d_text, d_joint)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        self.learning_rate = learning_rate

    def forward(self, images, texts):
        z_image = F.normalize(self.W_image(images), p=2, dim=1)
        z_text = F.normalize(self.W_text(texts), p=2, dim=1)
        return z_image, z_text

    def training_step(self, batch, batch_idx):
        images, texts = batch
        z_image, z_text = self(images, texts)
        
        # InfoNCE / CLIP loss
        logit_scale = torch.exp(self.temperature)
        logits_per_image = logit_scale * z_image @ z_text.T
        logits_per_text = logits_per_image.T
        
        labels = torch.arange(logits_per_image.shape[0], device=self.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        images, texts = batch
        z_image, z_text = self(images, texts)
        
        logit_scale = torch.exp(self.temperature)
        logits_per_image = logit_scale * z_image @ z_text.T
        logits_per_text = logits_per_image.T
        
        labels = torch.arange(logits_per_image.shape[0], device=self.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        
        # Calculate R@5 (dummy/approximate metric for demonstration inside batch)
        _, top_indices = logits_per_image.topk(5, dim=1)
        correct = (top_indices == labels.view(-1, 1)).any(dim=1)
        r_at_5 = correct.float().mean()
        
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_r_at_5', r_at_5, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


class MetDataModule(pl.LightningDataModule):
    """Lightning DataModule for the Met contrastive retrieval model.

    Parameters
    ----------
    images_tensor:
        Shape ``[N, d_image]`` — image embeddings for **all** items.
        Zero tensors are present for items without downloaded images.
    texts_tensor:
        Shape ``[N, d_text]`` — text embeddings for **all** items.
    has_image_mask:
        Boolean 1-D tensor of length N.  Only rows where this is ``True``
        are used during contrastive training, since InfoNCE requires genuine
        image-text pairs.
    batch_size:
        Mini-batch size for training and validation.
    """

    def __init__(
        self,
        images_tensor: torch.Tensor,
        texts_tensor: torch.Tensor,
        has_image_mask: torch.Tensor,
        batch_size: int = 256,
    ) -> None:
        super().__init__()
        self.images_tensor = images_tensor
        self.texts_tensor = texts_tensor
        self.has_image_mask = has_image_mask
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Restrict training pairs to items that have a real image.
        image_indices = self.has_image_mask.nonzero(as_tuple=False).squeeze(1)
        paired_images = self.images_tensor[image_indices]
        paired_texts = self.texts_tensor[image_indices]

        n = len(paired_images)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        test_size = n - train_size - val_size

        dataset = TensorDataset(paired_images, paired_texts)
        self.train_dataset, self.val_dataset, self.test_dataset = (
            torch.utils.data.random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)



class TelemetryCallback(pl.Callback):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 10 == 0:  # throttle updates
            loss = outputs['loss'].item() if isinstance(outputs, dict) else outputs.item()
            msg = {
                "type": "train_step",
                "epoch": trainer.current_epoch,
                "batch": batch_idx,
                "train_loss": loss
            }
            self.queue.put_nowait(json.dumps(msg))
            
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        val_r_at_5 = trainer.callback_metrics.get('val_r_at_5')
        
        msg = {
            "type": "val_epoch",
            "epoch": trainer.current_epoch,
            "val_loss": val_loss.item() if val_loss is not None else None,
            "val_r_at_5": val_r_at_5.item() if val_r_at_5 is not None else None
        }
        self.queue.put_nowait(json.dumps(msg))
