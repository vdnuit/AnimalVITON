# AnimalVITON
This repository is copy from the official implementation of OOTDiffusion

## Installation
1. Clone the repository

```sh
git clone https://github.com/vdnuit/AnimalVITON
```

2. Install the required packages

```sh
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
pip install ultralytics
pip install config einops onnxruntime diffusers==0.24.0 accelerate==0.26.1
```
3. Download checkpoints

The three types of checkpoints [OOTDiffusion Checkpoints](https://huggingface.co/levihsu/OOTDiffusion), [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14), [AnimalVITON_model](https://huggingface.co/skush1/AnimalVITON_model) should be placed under the OOTDiffusion/OOTDiffusion/checkpoints folder.
Below is the code for it.

```sh
git lfs install
git clone https://huggingface.co/levihsu/OOTDiffusion
git clone https://huggingface.co/openai/clip-vit-large-patch14
git clone https://huggingface.co/skush1/AnimalVITON_model
mv AnimalVITON_model/yolo-cloth OOTDiffusion/checkpoints/humanparsing OOTDiffusion/checkpoints/ootd OOTDiffusion/checkpoints/openpose clip-vit-large-patch14 AnimalVITON/OOTDiffusion/OOTDiffusion/checkpoints/
```
## Inference
> <model-image-path>: Path to the image of the dog that will virtually try on the clothes (e.g. examples/model/model_2.jpg)
> <cloth-image-path>: Path to the image of the dog clothing for virtual try-on (e.g. examples/garment/garment_1.jpg)

```sh
cd OOTDiffusion/OOTDiffusion/run
python run_ootd.py --model_path <model-image-path> --cloth_path <cloth-image-path> --scale 2.0 --sample 4
```
