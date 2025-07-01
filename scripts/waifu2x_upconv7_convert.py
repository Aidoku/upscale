import torch
import coremltools as ct
import numpy as np
import argparse
import os
import sys

# fix imports from nunif
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'nunif')))

from waifu2x.models import UpConv7
from nunif.models.utils import load_model

def main():
    parser = argparse.ArgumentParser(description="Convert a waifu2x .pth model to .mlpackage")
    parser.add_argument("pth_path", help="Path to the .pth model file")
    args = parser.parse_args()

    model, _ = load_model(args.pth_path)
    model.eval()

    size = 156
    example_input = torch.randn(1, 3, size, size)

    traced = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input", shape=example_input.shape, dtype=np.float32)
        ],
        outputs=[
            ct.TensorType(name="output")
        ],
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.iOS15
    )

    base = os.path.basename(args.pth_path)
    name, _ = os.path.splitext(base)
    mlmodel_name = f"waifu2x_{name}.mlpackage"
    mlmodel.save(mlmodel_name)
    print(f"Saved: {mlmodel_name}")

if __name__ == "__main__":
    main()
