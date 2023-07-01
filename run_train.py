import pytorch_lightning as pl
import torch

from image_dataset import ImageDataModule
from resnet50 import ResNet50


def main():
    torch.set_float32_matmul_precision('medium')

    # Define hyperparameters
    batch_size = 196
    num_epochs = 100
    learning_rate = 1e-3

    data_module = ImageDataModule("cifar10", batch_size, (224, 224))

    # Create an instance of the model
    n_class = len(data_module.dataset["train"].features["label"].names)
    model = ResNet50(n_class, learning_rate)

    # Create a trainer object with some settings
    trainer = pl.Trainer(max_epochs=num_epochs, default_root_dir="output")

    # Train the model on the training data and validate on the test data
    trainer.fit(model, data_module)
    result = trainer.test(model, data_module)
    print(result)


if __name__ == '__main__':
    main()
