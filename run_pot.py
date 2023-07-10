from pathlib import Path

import nncf
import openvino.runtime as ov
from openvino.tools import mo

from image_dataset import ImageDataModule
from resnet50 import ResNet50


def convert_torch_to_openvino(cpt_path: str, input_shape) -> None:
    model = ResNet50.load_from_checkpoint(cpt_path).model
    model = mo.convert_model(model, input_shape=input_shape)
    model_xml = "output/model/ov/fp32/model.xml"
    Path(model_xml).parent.mkdir(parents=True, exist_ok=True)
    ov.serialize(model, model_xml)


def quantization(model_fp32_xml: str) -> None:
    data_module = ImageDataModule("cifar10", 1, (224, 224))
    data_module.prepare_data()
    data_module.setup("fit")
    dataloader = data_module.train_dataloader()
    dataset = nncf.Dataset(dataloader, lambda item: item[0].numpy())

    model_int8_xml = "output/model/ov/int8/model.xml"
    model_fp32 = ov.Core().read_model(model_fp32_xml)
    model_int8 = nncf.quantize(model_fp32, dataset, subset_size=len(dataloader))
    ov.serialize(model_int8, model_int8_xml)


def main():
    path = "output/lightning_logs/version_25/checkpoints/ep=19-acc=0.856.ckpt"
    convert_torch_to_openvino(path, (1, 3, 224, 224))

    path = "output/model/ov/fp32/model.xml"
    quantization(path)


if __name__ == '__main__':
    main()
