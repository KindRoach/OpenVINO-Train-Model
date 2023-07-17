from statistics import mean

import openvino.runtime as ov
import torch
from pytorch_lightning import Trainer
from torchmetrics.functional import accuracy
from tqdm import tqdm

from image_dataset import load_datamodule
from resnet50 import ResNet50


def main():
    data_module = load_datamodule("cifar10", 128)
    torch_model_path = "output/lightning_logs/version_25/checkpoints/ep=19-acc=0.856.ckpt"
    torch_acc = benchmark_torch_model(data_module, torch_model_path)

    ov_model_path = "output/model/ov/fp32/model.xml"
    ov_fp32_acc = benchmark_ov_model(data_module, ov_model_path)

    ov_model_path = "output/model/ov/int8-pot/model.xml"
    ov_int8_pot_acc = benchmark_ov_model(data_module, ov_model_path)

    ov_model_path = "output/model/ov/int8-qac/model.xml"
    ov_int8_qac_acc = benchmark_ov_model(data_module, ov_model_path)

    print(
        f"torch_acc={torch_acc:.3f}\n"
        f"ov_fp32_acc={ov_fp32_acc:.3f}\n"
        f"ov_int8_pot_acc={ov_int8_pot_acc:.3f}\n"
        f"ov_int8_qac_acc={ov_int8_qac_acc:.3f}"
    )


def benchmark_torch_model(data_module, torch_model_path):
    torch.set_float32_matmul_precision('medium')
    model = ResNet50.load_from_checkpoint(torch_model_path)
    trainer = Trainer()
    result = trainer.test(model, data_module, verbose=False)
    return result[0]["test_acc"]


def benchmark_ov_model(data_module, ov_model_path):
    ie = ov.Core()
    model = ie.read_model(ov_model_path)
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
