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

## Inference
```sh
cd OOTDiffusion/OOTDiffusion/run
python run_ootd.py --model_path examples/model/model_2.jpg --cloth_path examples/garment/garment_1.jpg --scale 2.0 --sample 4
```
