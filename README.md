# SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation

[Hao Shi](https://shihao1895.github.io/), [Bin Xie](https://xb534.github.io/), [Yingfei Liu](https://scholar.google.com/citations?user=pF9KA1sAAAAJ), [Yang Yue](https://yueyang130.github.io), [Tiancai Wang](https://scholar.google.com/citations?user=YI0sRroAAAAJ), [Haoqiang Fan](https://scholar.google.com/citations?user=bzzBut4AAAAJ), [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ), [Gao Huang](https://scholar.google.com/citations?user=-P9LwcgAAAAJ)

Tsinghua University, Dexmal, MEGVII, StepFun

AAAI 2026 **Oral**

> This is the code for the paper "SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation".

### 🏠[Project Page](https://shihao1895.github.io/SpatialActor/) | 📑[Paper](https://arxiv.org/abs/2511.09555) | 🤗[HuggingFace](https://huggingface.co/collections/shihao1895/spatialactor)

## 🌟 News

- 🔥 [2026-1-8] The code of [SpatialActor](https://arxiv.org/abs/2511.09555) is released!

- 🔥 [2025-11-8] Our paper [SpatialActor](https://arxiv.org/abs/2511.09555) is accepted as an AAAI 2026 **Oral**!

## Overview

SpatialActor is a disentangled framework for robust robotic manipulation. It decouples perception into complementary high-level geometry from fine-grained but noisy raw depth and coarse but robust depth expert priors, along with low-level spatial cues and appearance semantics.

![SpatialActor Overview](images/fig_overall.png)

## Install

**Step 1:** Install Python, Pytorch and CUDA.

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to set up an environment. The code is built using Python 3.9, and we use PyTorch == 1.12.1 and CUDA == 11.3 (It may run with other versions, but we have not tested it). More instructions to install PyTorch can be found [here](https://pytorch.org/).

```bash
conda create --name spact python=3.9
conda activate spact

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

conda install -c "nvidia/label/cuda-11.3.1" cudatoolkit=11.3 -y
```

**Step 2:** Install PyTorch3D and XFormers.

we use pytorch3d == 0.7.5, xformers == 0.0.16

```bash
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu113_pyt1121.tar.bz2

conda install https://anaconda.org/xformers/xformers/0.0.16/download/linux-64/xformers-0.0.16-py39_cu11.3_pyt1.12.1.tar.bz2
```

**Step 3:** Install CoppeliaSim.

PyRep requires version **4.1** of CoppeliaSim. Download and unzip CoppeliaSim:

```bash
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu20_04.tar.xz --no-check-certificate

tar -xvf CoppeliaSim_Player_V4_1_0_Ubuntu20_04.tar.xz
```

- [CoppeliaSim-V4.1-Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz)

- [CoppeliaSim-V4.1-Ubuntu 20.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, please modify the path, and add the following to your `~/.bashrc` file.

```bash
export COPPELIASIM_ROOT=/PATH/TO/COPPELIASIM/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
```

Remember to source your .bashrc (`source ~/.bashrc`) after this.

**Step 4:** Install required libraries, including PyRep, YARR, RLBench and Point Renderer.

Git clone and install these required libraries.

```bash
pip install -e .

cd third_libs
git clone https://github.com/stepjam/PyRep.git
git clone https://github.com/NVlabs/YARR.git
git clone https://github.com/buttomnutstoast/RLBench.git

pip install -e PyRep
pip install -e YARR
pip install -e RLBench
pip install -e point-renderer
cd ..
pip install -e .
```

## Training

**Step1**: Download dataset and pretrained model.

- Download [RLBench](https://huggingface.co/datasets/shihao1895/rlbench) dataset, following [PerAct](https://github.com/peract/peract#download) setup. (~70 GB)

- Download [DepthAnythingV2 model](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true), and modify `model.dep_exp_path` in `spatial_actor/configs/spact.yaml`. (~100 MB)

- (Optional but **recommend**) Download [Replay](https://huggingface.co/datasets/shihao1895/spact-rlbench-replay) to reduce the startup time. (~85 GB)

  > We use the same dataloader as PerAct, which is based on [YARR](https://github.com/stepjam/YARR). YARR creates a replay buffer on the fly which can reduce the startup time.

**Step2**: Training SpatialActor.

To train SpatialActor on all RLBench tasks, use the following command (from `script/train.sh`):
```bash
python spatial_actor/train.py \
  --device 0,1,2,3,4,5,6,7 \
  --iter-based \
  --data-folder YOUR_DATA_FOLDER \
  --train-replay-dir YOUR_REPLAY_DIR \
  --log-dir YOUR_LOG_DIR \
  --cfg_path spatial_actor/configs/spact.yaml \
  --cfg_opts ""
```

- The configs can be overwritten by `cfg_opts` string of format `<cfg1> <val1> <cfg2> <val2> ..`.
- For training on 18 RLBench tasks, with 100 demos per task, we use 8 RTX4090 or A100 GPUs. The model trains in ~29 h. 

## Evaluation

**Step1** (Optional): Prepare the headless environment.

If you run evaluation on a headless server, you need install a virtual desktop.

```bash
apt update
apt install xfce4 xvfb
apt install libnvidia-gl-535 # Modified by your nvidia driver version
```

If you get qt plugin error like `qt.qpa.plugin: Could not load the Qt platform plugin "xcb" <somepath>/cv2/qt/plugins" even though it was found`, try uninstalling opencv-python and installing opencv-python-headless.

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

Then, start the virtual desktop.

```bash
nohup Xvfb :33 -screen 0 640x360x24 &
startxfce4 &
export DISPLAY=:33
```

If you have other problems, you can refer to [RLBench Repo](https://github.com/stepjam/RLBench).

**Step2**: Evaluate SpatialActor on RLBench

Run evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 \
python spatial_actor/eval.py \
    --eval-datafolder YOUR_DATA_FOLDER/test \
    --model-path YOUR_MODEL_PATH \
    --tasks all \
    --device 0 \
    --eval-episodes 25 \
    --log-name rlbench_all \
    --headless
```
## TODO

All components are now available.

- [x] Code Release
- [x] Model Weights Release
- [x] Dataset Upload to HuggingFace

**NOTE:** 

1. The original experiments were conducted on NVIDIA A100 GPUs. For easier reproduction and broader accessibility, we adapted the codebase to run on RTX 4090 GPUs with minor configuration changes. Consequently, the results may not be strictly identical to the original A100-based results.
2. RLBench evaluation has some stochasticity, and evaluating multiple epoch checkpoints (e.g., 35, 40, 45, 50) is recommended.

## Acknowledgement

We sincerely thank the authors of [RVT](https://github.com/nvlabs/rvt), [PerAct](https://github.com/peract/peract), [PyRep](https://github.com/stepjam/PyRep), [RLBench](https://github.com/stepjam/RLBench), [YARR](https://github.com/stepjam/YARR) for sharing their code.

## Citation

If you find our work helpful in your research, please consider citing [our paper](https://arxiv.org/abs/2511.09555). 

```bibtex
@article{shi2025spatialactor,
  title={SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation},
  author={Shi, Hao and Xie, Bin and Liu, Yingfei and Yue, Yang and Wang, Tiancai and Fan, Haoqiang and Zhang, Xiangyu and Huang, Gao},
  journal={arXiv preprint arXiv:2511.09555},
  year={2025}
}
```
