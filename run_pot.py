from pathlib import Path
from statistics import mean

import nncf
import openvino.runtime as ov
import torch
import tqdm
from openvino.tools import mo
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from tqdm import tqdm

from image_dataset import load_datamodule
from resnet50 import ResNet50


def convert_torch_to_openvino(cpt_path: str, input_shape) -> None:
    model = ResNet50.load_from_checkpoint(cpt_path).model
    model = mo.convert_model(model, input_shape=input_shape)

    model_xml = "output/model/ov/fp32/model.xml"
    Path(model_xml).parent.mkdir(parents=True, exist_ok=True)

    change_to_dynamic_shape(model)
    ov.serialize(model, model_xml)


def change_to_dynamic_shape(model):
    # convert to dynamic shape
    shapes = {}
    for input_layer in model.inputs:
        shapes[input_layer] = input_layer.partial_shape
        shapes[input_layer][0] = -1
    model.reshape(shapes)


def pot_quantization(model_fp32_xml: str) -> None:
    data_module = load_datamodule("cifar10", 8)
    data_module.prepare_data()
    data_module.setup("fit")
    dataloader = data_module.train_dataloader()
    dataset = nncf.Dataset(dataloader, lambda item: item[0].numpy())

    model_int8_xml = "output/model/ov/int8-pot/model.xml"
    model_fp32 = ov.Core().read_model(model_fp32_xml)
    model_int8 = nncf.quantize(model_fp32, dataset, subset_size=len(dataloader))

    change_to_dynamic_shape(model_int8)
    ov.serialize(model_int8, model_int8_xml)


def qac_quantization(model_fp32_xml: str) -> None:
    data_module = load_datamodule("cifar10", 8)
    data_module.prepare_data()
    data_module.setup("fit")

    dataloader = data_module.train_dataloader()
    train_dataset = nncf.Dataset(dataloader, lambda item: item[0].numpy())
    dataloader = data_module.val_dataloader()
    val_dataset = nncf.Dataset(dataloader, lambda item: item[0].numpy())

    def validate(model: ov.CompiledModel, validation_loader: DataLoader) -> float:
        n_class = len(data_module.dataset["train"].features["label"].names)
        all_acc = []
        infer_req = model.create_infer_request()
        for inputs, labels in tqdm(validation_loader):
            infer_req.infer(inputs)
            outputs = infer_req.get_output_tensor().data
            outputs = torch.tensor(outputs)
            acc = accuracy(outputs, labels, task="multiclass", num_classes=n_class)
            all_acc.append(acc.item())
        return mean(all_acc)

    model_int8_xml = "output/model/ov/int8-qac/model.xml"
    model_fp32 = ov.Core().read_model(model_fp32_xml)
    model_int8 = nncf.quantize_with_accuracy_control(
        model_fp32,
        calibration_dataset=train_dataset,
        validation_dataset=val_dataset,
        validation_fn=validate)

    change_to_dynamic_shape(model_int8)
    ov.serialize(model_int8, model_int8_xml)


def main():
    path = "output/lightning_logs/version_25/checkpoints/ep=19-acc=0.856.ckpt"
    convert_torch_to_openvino(path, (1, 3, 224, 224))

    path = "output/model/ov/fp32/model.xml"
    pot_quantization(path)
    qac_quantization(path)


if __name__ == '__main__':
    main()
