<p align="center">
  <img src="assets/logo.png" height=120>
</p>

# <p align="center"> [NeurIPS 2025] DIPO: Dual-State Images Controlled Articulated Object Generation Powered by Diverse Data </p>


![Python 3.10](https://img.shields.io/badge/python-3.8-g) ![pytorch 2.3.1](https://img.shields.io/badge/pytorch-2.3.1-blue.svg)

:rocket: This repository is the official implementation of [DIPO](https://arxiv.org/pdf/2505.20460), which is a framework that generate articulated objects conditioned on **Dual-State Image Pairs** (resting and articulated states)

> **[NeruIPS 2025] DIPO: Dual-State Images Controlled Articulated Object Generation Powered by Diverse Data**<br>
> Ruiqi Wu, Xinjie Wang, Liu Liu, Chunle Guo*, Jiaxiong Qiu, Chongyi Li, Lichao Huang, Zhizhong Su, Ming-Ming Cheng
><br>( * indicates corresponding author)

[[Arxiv Paper](https://arxiv.org/pdf/2505.20460)]&nbsp;
[[中文版](https://rq-wu.github.io/projects/DIPO/DIPO_CN.pdf)]&nbsp;
[[Website Page](https://rq-wu.github.io/projects/DIPO/index.html)]&nbsp;
[[PM-X (dataset)](https://huggingface.co/datasets/HorizonRobotics/DIPO-Dataset)]&nbsp;
[[Gradio Demo](https://huggingface.co/spaces/HorizonRobotics/DIPO)]&nbsp;

<img src="assets/method.png">

## Preparation
### Dependencies and Installation
We recommend to use [miniconda](https://docs.anaconda.com/miniconda/) to manage the environment. The environment was tested on Ubuntu 20.04.4 LTS.
```
# Create a conda environment
conda create -n dipo python=3.10
conda activate dipo

# Install Pytorch
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other packages
pip install -r requirement.txt

# Install Pytorch3D (for evaluation)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### GPT-4o Settings
input your key of GPT-4o in `scripts/graph_pred/api.py`.
```
client = AzureOpenAI(
    azure_endpoint="your_endpoint",
    api_key="your_key",
    api_version="your_version",
)
```

## Download

### PM-X Dataset
Our PM-X dataset is constructed by an agent system, named LEGO-Art. It builds complex articulated objects with primitives provieded by Partnet-Mobility dataset. You can download the novel dataset at [link]().
<img src="assets/PM-X.png">

### PM + ACD Dataset
You can download the origin data and our proprocessed data from [here](), for training and evaluation.

### Checkpoints
You can download DIPO checkpoint file for inference and CAGE pre-trained weights for training from [here]().
```
<project directory>
├── ckpts
│   ├── cage_cfg.ckpt
│   ├── dipo.ckpt
```

### 3D assets for mesh retrieval
Download 3D assets for mesh retrieval from [here](https://huggingface.co/datasets/wuruiqi0722/DIPO_data), which also the original data of a subset of PartNet-Mobility Dataset.


## Usage
### Quick Demo
We provide a quick demo to run the inference on a dual-state image pair. 
```
python demo_img.py \
--configs/config.yaml \
--ckpt_path ckpts/dipo.ckpt \
--img_path_1 path/of/the/resting/state/image \
--img_path_2 path/of/the/articulated/state/image
```

If you successfully run the script, the output will be saved at `./results`. By default, there will be three objects generated out by initializing with different noises.
For other configuration, please see the arguments in the script.

### Evaluation
If you're interested in evaluating our model on the test set (see the data split in `data/data_split.json` for PartNet-Mobility, and in `data/data_acd.json` for ACD dataset), you can run the test script as below. 
```
# Evaluate on the test set (given GT graph, no object category label)
python test.py \
    --config configs/config.yaml \
    --ckpt ckpts/dipo.ckpt \ 
    --label_free \
    --which_data pm
```
The evaluation is only supported on a single GPU, which was tested on a NVIDIA 4090 (24GB).

### Training
We train our model on top of a [CAGE](https://3dlg-hcvc.github.io/cage/) model pretrained under our setting. This checkpoint can be downloaded [here](https://aspis.cmpt.sfu.ca/projects/singapo/ckpts/pretrained_cage.zip), which is put under `pretrained` folder by default.
```
<project directory>
├── pretrained
│   ├── cage_cfg.ckpt
```
Run the following command to train our model from scratch. The original model is trained on 4 NVIDIA A100s.
```
python train.py \
    --config configs/config.yaml \
    --pretrained_cage ckpts/cage_cfg.ckpt
```

### LEGO-Art Pipeline
```
# Step-1: Roll description & Build grid-level data
python scripts/layout_generator/api.py --save_path path/to/gpt/data --obj_num 3

# Step-2: Build data with coordinates
python scripts/layout_generator/layout_generator_in_grid.py --save_path path/to/gpt/data

# Step-3 Retrival
python scripts/mesh_retrieval/retrieval.py --src_dir path/to/gpt/data --gt_data_root path/to/assets/for/retrieval

# Step-4 Render data with Blender
python scripts/render/render_dir.py --src_dir path/to/gpt/data

# Step-5 Filter data with VLMs
python scripts/layout_generator/api_filter.py --save_path path/to/gpt/data
```


## Citation
```
@inproceedings{wu2025dipo,
  title={DIPO: Dual-State Images Controlled Articulated Object Generation Powered by Diverse Data},
  author={Wu, Ruqi and Wang, Xinjie and Liu, Liu and Guo, Chunle and Qiu, Jiaxiong and Li, Chongyi and Huang, Lichao and Su, Zhizhong and Cheng, Ming-Ming},
  booktitle={Advances in Neural Information Processing Systems 39 (NeurIPS 2025)},
  year={2025}
}
```
