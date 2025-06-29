import torch
import coremltools as ct
import numpy as np
import argparse
import os
from spandrel import MAIN_REGISTRY, ModelLoader, ImageModelDescriptor

def main():
    parser = argparse.ArgumentParser(description="Convert a .pth model to .mlpackage with spandrel")
    parser.add_argument("pth_path", help="Path to the .pth model file")
    parser.add_argument("--scale", help="Model scale factor", type=int, default=4)
    args = parser.parse_args()

    model = ModelLoader().load_from_file(args.pth_path).model
    model.eval()

    size = 256
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
    )

    base = os.path.basename(args.pth_path)
    name, _ = os.path.splitext(base)
    mlmodel_name = f"{name}.mlpackage"
    mlmodel.save(mlmodel_name)
    print(f"Saved: {mlmodel_name}")

if __name__ == "__main__":
    main()
