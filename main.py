import os
import pytorch_lightning as pl
# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torchvision.models import mobilenet_v2

from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import ImageFolder

import wandb



class CustomImageFolderDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = 'paddy-disease-classification/train_images'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Augmentation for training
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Basic transform for val/test
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage=None):
        # Load full dataset with no transform first
        full_dataset = ImageFolder(root=self.data_dir)

        # Split indices
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        # Random split with fixed seed
        train_data, val_data, test_data = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Apply transform manually per split
        train_data.dataset.transform = self.train_transform
        val_data.dataset.transform = self.test_transform
        test_data.dataset.transform = self.test_transform

        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = test_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    

class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })


class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Load pretrained MobileNetV2
        self.model = mobilenet_v2(pretrained=True)

        # Thay thế classifier cuối cùng: MobileNetV2 có classifier là nn.Sequential với 2 layer
        # Layer cuối cùng là nn.Linear với in_features = 1280 (đã cố định trong mobilenet_v2)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    # returns the size of the output tensor going into Linear layer from the conv block.

    # returns the feature tensor from the conv block

    # will be used during inference
    def forward(self, x):
        return self.model(x)

    
  
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)  # ✅ dùng cross_entropy

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss
        

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

if __name__=="__main__":

    dm = CustomImageFolderDataModule(batch_size=64)
    # To access the x_dataloader we need to call prepare_data and setup.
    dm.prepare_data()
    dm.setup()

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(dm.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]
    val_imgs.shape, val_labels.shape

    model = LitModel((3, 32, 32), 10)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

    # Initialize Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss",patience=10)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="mobilenet-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_weights_only=False
    )




    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=200,
                        devices=[0], 
                        logger=wandb_logger,
                        callbacks=[early_stop_callback,
                                    ImagePredictionLogger(val_samples),
                                    checkpoint_callback],strategy="auto"
                                    # accelerator="auto"
                        )

    # Train the model ⚡🚅⚡
    trainer.fit(model, dm)

    # Evaluate the model on the held-out test set ⚡⚡
    trainer.test(dataloaders=dm.test_dataloader())

    # Close wandb run
    wandb.finish()