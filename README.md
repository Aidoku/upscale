# Aidoku Image Upscaling

This repository contains image upscaling models that are installable within Aidoku.

## Available Models

- `waifu2x (photo, 2x, noise0)`: the latest version of waifu2x, with 2x scaling and noise0 denoising.
- `Real-ESRGAN (photo, 2x)`: Real-ESRGAN 2x model for general images.

## Using Custom Models in Aidoku

Aidoku supports custom CoreML models by placing them in the `Models` directory in Aidoku's documents. You can follow these steps:

1. Obtain a `.mlmodel` file or `.mlpackage` folder (see [scripts](scripts/) for converting your own).
2. Create a `.json` file with the same name as your model file (e.g. `test.mlmodel` would have a corresponding `test.mlmodel.json` file)
	- See the `json` files in the [models](models/) directory for examples.
	- The `type` value should be `image` or `multiarray`, depending on the model type.
	- `multiarray` models should be configured with the input and output names as well as the block size.
3. Place both the model and json file in the `Models` directory in the Aidoku documents directory.
4. Select the model from the upscaling models page in Aidoku.
