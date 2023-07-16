import sys
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from simple_parsing import ArgumentParser

from image_dataset import ImageDataModule, load_datamodule
from resnet50 import ResNet50, Args


def main(args: Args):
    torch.set_float32_matmul_precision('medium')
    data_module = load_datamodule("cifar10", args.batch_size)
    model = ResNet50(**args.__dict__)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10, monitor="val_acc", mode="max",
        auto_insert_metric_name=False,
        filename="ep={epoch}-acc={val_acc:.3f}"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc", mode="max",
        min_delta=0.00, patience=5,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        default_root_dir="output",
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(model, data_module)
    trainer.test(model, data_module, verbose=True)
    print(f"best model saved as: {checkpoint_callback.best_model_path}")


def parse_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="arguments")
    return parser.parse_args(args).arguments


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
