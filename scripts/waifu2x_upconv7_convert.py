import torch
import coremltools as ct
import numpy as np
import argparse
import json
import os
import sys

# fix imports from nunif
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "nunif")))

from waifu2x.models import UpConv7
from nunif.models.utils import load_model

_ = UpConv7  # prevent unused warning

SOURCE_DIR = "./pretrained_models/upconv_7"
OUTPUT_DIR = "./aidoku_models"

BLOCK_SIZE = 156
SHRINK_SIZE = 7
INPUT_NAME = "input"
OUTPUT_NAME = "output"


def infer_tags_and_name(file_name, model_type):
    base = os.path.splitext(file_name)[0]
    tags = []
    if "scale2x" in base:
        tags.append("2x")
    else:
        tags.append("1x")
    for noise in ["noise0", "noise1", "noise2", "noise3"]:
        if noise in base:
            tags.append(noise)
    tags_name = tags.copy()
    tags.append("fast")
    tags_name.insert(0, model_type)

    name = f"waifu2x ({', '.join(tags_name)})"
    return tags, name


def convert_model(pth_path, output_dir, model_type):
    print(f"Converting {pth_path} ({model_type})...")
    model, _ = load_model(pth_path)
    model.eval()

    example_input = torch.randn(1, 3, BLOCK_SIZE, BLOCK_SIZE)
    traced = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name=INPUT_NAME, shape=example_input.shape, dtype=np.float32)
        ],
        outputs=[ct.TensorType(name=OUTPUT_NAME)],
        convert_to="neuralnetwork",
        minimum_deployment_target=ct.target.iOS13,
    )
    mlmodel_fp16 = ct.models.neural_network.quantization_utils.quantize_weights(
        mlmodel, nbits=16
    )

    base = os.path.splitext(os.path.basename(pth_path))[0]
    mlmodel_file_name = f"waifu2x_{model_type}_{base}.mlmodel"
    mlmodel_path = os.path.join(output_dir, mlmodel_file_name)
    mlmodel_fp16.save(mlmodel_path)

    tags, human_name = infer_tags_and_name(base, model_type)
    json_data = {
        "name": human_name,
        "info": "A lightweight image 2x upscale and denoise model.",
        "tags": tags,
        "type": "multiarray",
        "config": {
            "inputName": INPUT_NAME,
            "outputName": OUTPUT_NAME,
            "blockSize": BLOCK_SIZE,
            "shrinkSize": SHRINK_SIZE,
        },
        "file": mlmodel_file_name,
    }
    if "1x" in tags:
        json_data["config"]["scale"] = 1

    json_path = os.path.join(output_dir, f"{mlmodel_file_name}.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"Saved: {mlmodel_file_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert waifu2x .pth models to .mlmodels"
    )
    parser.add_argument(
        "input_dir",
        help="Path to the art and photo subdirectories containing upconv7 models.",
        default=SOURCE_DIR,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Path to where converted models should be output.",
        default=OUTPUT_DIR,
    )
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(args.output_dir)

    for model_type in ["art", "photo"]:
        folder = os.path.join(args.input_dir, model_type)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith(".pth"):
                pth_path = os.path.join(folder, file)
                convert_model(pth_path, args.output_dir, model_type)

    print("All models converted!")


if __name__ == "__main__":
    main()
