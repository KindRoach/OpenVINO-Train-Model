import torch
from statistics import mean

import openvino.runtime as ov
from pytorch_lightning import Trainer
from torchmetrics.functional import accuracy
from tqdm import tqdm

from image_dataset import ImageDataModule
from resnet50 import ResNet50


def main():
    data_module = ImageDataModule("cifar10", 128, (224, 224))
    ov_model_path = "output/model/ov/fp32/model.xml"
    torch_model_path = "output/lightning_logs/version_25/checkpoints/ep=19-acc=0.856.ckpt"
    torch_acc = benchmark_torch_model(data_module, torch_model_path)
    ov_acc = benchmark_ov_model(data_module, ov_model_path)
    print(f"torch_acc={torch_acc:.3f}\nov_acc={ov_acc:.3f}")


def benchmark_torch_model(data_module, torch_model_path):
    torch.set_float32_matmul_precision('medium')
    model = ResNet50.load_from_checkpoint(torch_model_path)
    trainer = Trainer()
    result = trainer.test(model, data_module, verbose=False)
    return result[0]["test_acc"]


def benchmark_ov_model(data_module, ov_model_path):
    ie = ov.Core()
    model = ie.read_model(ov_model_path)
    shapes = {}
    for input_layer in model.inputs:
        shapes[input_layer] = input_layer.partial_shape
        shapes[input_layer][0] = -1
    model.reshape(shapes)
    compiled_model = ie.compile_model(model)
    data_loader = data_module.test_dataloader()
    n_class = len(data_module.dataset["train"].features["label"].names)
    all_acc = []
    infer_req = compiled_model.create_infer_request()
    for inputs, labels in tqdm(data_loader):
        infer_req.infer(inputs)
        outputs = infer_req.get_output_tensor().data
        outputs = torch.tensor(outputs)
        acc = accuracy(outputs, labels, task="multiclass", num_classes=n_class)
        all_acc.append(acc.item())
    return mean(all_acc)


if __name__ == '__main__':
    main()