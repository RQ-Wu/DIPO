<p align="center">
  <img src="assets/logo.png" height=120>
</p>

# <p align="center"> DIPO: Dual-State Images Controlled Articulated Object Generation Powered by Diverse Data </p>


![Python 3.10](https://img.shields.io/badge/python-3.8-g) ![pytorch 2.3.1](https://img.shields.io/badge/pytorch-2.3.1-blue.svg)

:rocket: This repository is the official implementation of [DIPO](https://arxiv.org/pdf/2505.20460), which is a framework that generate articulated objects conditioned on **Dual-State Image Pairs** (resting and articulated states)

> **DIPO: Dual-State Images Controlled Articulated Object Generation Powered by Diverse Data**<br>
> Ruiqi Wu, Xinjie Wang, Liu Liu, Chunle Guo*, Jiaxiong Qiu, Chongyi Li, Lichao Huang, Zhizhong Su, Ming-Ming Cheng
><br>( * indicates corresponding author)

[[Arxiv Paper](https://arxiv.org/pdf/2505.20460)]&nbsp;
[[Website Page](https://rq-wu.github.io/projects/LAMP/index.html)]&nbsp;
[[PM-X (dataset)]()]&nbsp;
[[Gradio Demo](https://colab.research.google.com/drive/1Cw2e0VFktVjWC5zIKzv2r7D2-4NtH8xm?usp=sharing)]&nbsp;

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
<!-- We also share the graph prediction results [here](https://aspis.cmpt.sfu.ca/projects/singapo/pred_graph.zip) so that you can run the evaluation by taking the graph prediction from GPT-4o as input. Once downloaded, you can put it under the `exps` directory, as shown in the following file structure.
```
<project directory>
├── exps
│   ├── predict_graph
│   │   ├── acd_test
│   │   ├── pm_test
``` -->
<!-- To use these recordings of the graph prediction for evaluation, you need to specify the path to one of the prediction folders `--G_dir`. For example,
```
# Evaluate on the test set (given predicted graph, no object category label)
python test.py \
    --config exps/singapo/final/config/parsed.yaml \
    --ckpt exps/singapo/final/ckpts/last.ckpt \
    --label_free \
    --which_data pm \ 
    --G_dir exps/pred_graph/pm_test
``` -->
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
    --pretrained_cage pretrained/cage_cfg.ckpt
```

## Citation
```
@article{wu2025dipo,
  title={DIPO: Dual-State Images Controlled Articulated Object Generation Powered by Diverse Data},
  author={Wu, Ruqi and Wang, Xinjie and Liu, Liu and Guo, Chunle and Qiu, Jiaxiong and Li, Chongyi and Huang, Lichao and Su, Zhizhong and Cheng, Ming-Ming},
  journal={arXiv preprint arXiv:2505.20460},
  year={2025}
}
```