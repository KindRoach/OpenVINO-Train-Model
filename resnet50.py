import pytorch_lightning as pl
import torch
import torchvision
from torch.nn.functional import cross_entropy
from torchmetrics.functional import accuracy


class ResNet50(pl.LightningModule):

    def __init__(self, n_class: int, lr: float):
        super().__init__()
        self.lr = lr
        self.n_class = n_class
        # Load the non-pretrained resnet-50
        self.model = torchvision.models.resnet50()
        # Replace the last fully connected layer
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, n_class)

    def forward_loss(self, batch):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = cross_entropy(outputs, labels)
        acc = accuracy(outputs, labels, task="multiclass", num_classes=self.n_class)
        return acc, loss

    def training_step(self, batch, batch_idx):
        acc, loss = self.forward_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        acc, loss = self.forward_loss(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        acc, loss = self.forward_loss(batch)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
