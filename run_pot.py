from pathlib import Path

import openvino.runtime as ov
from openvino.tools import mo

from resnet50 import ResNet50


def convert_torch_to_openvino(cpt_path: str, input_shape) -> None:
    model = ResNet50.load_from_checkpoint(cpt_path).model
    model = mo.convert_model(model, input_shape=input_shape)
    model_xml = "output/model/ov/fp32/model.xml"
    Path(model_xml).parent.mkdir(parents=True, exist_ok=True)
    ov.serialize(model, model_xml)


def main():
    path = "output/lightning_logs/version_25/checkpoints/ep=19-acc=0.856.ckpt"
    convert_torch_to_openvino(path, (1, 3, 224, 224))


if __name__ == '__main__':
    main()
