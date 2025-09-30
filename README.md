<div align="center">

# _FaceXFormer_ : A Unified Transformer <br> for Facial Analysis
<h3><strong>ICCV 2025</strong></h3>

[Kartik Narayan*](https://kartik-3004.github.io/portfolio/) &emsp; [Vibashan VS*](https://vibashan.github.io) &emsp; [Rama Chellappa](https://engineering.jhu.edu/faculty/rama-chellappa/) &emsp; [Vishal M. Patel](https://engineering.jhu.edu/faculty/vishal-patel/)  

Johns Hopkins University

<a href='https://kartik-3004.github.io/facexformer/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/2403.12960v3'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/kartiknarayan/facexformer'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>

</div>

Official implementation of **[_FaceXFormer_ : A Unified Transformer for Facial Analysis](https://kartik-3004.github.io/facexformer/)**.
<hr />

## Highlights

**_FaceXFormer_**, is the first unified transformer for facial analysis:

1️⃣ that is capable of handling a comprehensive range of facial analysis tasks such as face parsing, landmark detection, head pose estimation, attributes recognition, age/gender/race estimation, facial expression recognition, and face visibility prediction<br>
2️⃣ that leverages a transformer-based encoder-decoder architecture where each task is treated as a learnable token, enabling the integration of multiple tasks within a single framework<br>
3️⃣ that effectively handles images "in-the-wild," demonstrating its robustness and generalizability across nine heterogenous tasks, all while maintaining the real-time performance of 33.21 FPS<br>

<p align="center" width="100%">
  <img src='docs/static/images/intro.png'>
</p>

> **<p align="justify"> Abstract:** *In this work, we introduce <i>FaceXFormer</i>, an end-to-end unified 
transformer model capable of performing nine facial analysis tasks including face parsing, landmark detection, 
head pose estimation, attribute prediction, and estimation of age, gender, race, expression, and face visibility 
within a single framework. Conventional methods in face analysis have often relied on task-specific designs 
and pre-processing techniques, which limit their scalability and integration into a unified architecture. 
Unlike these conventional methods, <i>FaceXFormer</i> leverages a transformer-based encoder-decoder architecture 
where each task is treated as a learnable token, enabling the seamless integration and simultaneous processing of 
multiple tasks within a single framework. Moreover, we propose a novel parameter-efficient decoder, FaceX, which 
jointly processes face and task tokens, thereby learning generalized and robust face representations across 
different tasks. We jointly trained <i>FaceXFormer</i> on nine face perception datasets and conducted experiments 
against specialized and multi-task models in both intra-dataset and cross-dataset evaluations across multiple benchmarks, showcasing state-of-the-art or competitive performance. Further, we performed a comprehensive analysis of different 
backbones for unified face task processing and evaluated our model in-the-wild, demonstrating its robustness and generalizability. To the best of our knowledge, this is the first work to propose a single model capable of 
handling nine facial analysis tasks while maintaining real-time performance at 33.21 FPS.* </p>

# :rocket: News
- [03/19/2024] 🔥 We release <i>FaceXFormer</i>.

## Installation

### Using pip (recommended)
```bash
# Clone the repository
git clone https://github.com/ernestchu/facexformer.git
cd facexformer

# Install dependencies
pip install -r requirements.txt
```

### Using conda (optional)
If you prefer using conda for environment management:
```bash
# Create a new conda environment
conda create -n facexformer python=3.10
conda activate facexformer

# Install dependencies
pip install -r requirements.txt
```

**Note:** PyTorch will be installed automatically. If you need a specific CUDA version, install PyTorch separately first:
```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
## Download Models

The model weights will be **automatically downloaded** from [HuggingFace](https://huggingface.co/kartiknarayan/facexformer) on first use.

Alternatively, you can download them manually:
```bash
# Using Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="kartiknarayan/facexformer", filename="ckpts/model.pt", local_dir="./")
```

Or download directly from the [HuggingFace repository](https://huggingface.co/kartiknarayan/facexformer).
## Usage

The model automatically downloads weights from [HuggingFace](https://huggingface.co/kartiknarayan/facexformer) on first use.

### Basic Usage

```bash
# Face parsing
python inference.py --image image.png --task parsing

# Landmark detection
python inference.py --image image.png --task landmarks

# Head pose estimation
python inference.py --image image.png --task headpose

# Face attributes
python inference.py --image image.png --task attributes

# Age, gender, and race estimation
python inference.py --image image.png --task age_gender_race

# Landmark visibility
python inference.py --image image.png --task visibility
```

### Advanced Options

```bash
# Custom output directory
python inference.py --image image.png --task parsing --output-dir ./my_results

# Use specific model path (skip auto-download)
python inference.py --image image.png --task parsing --model-path ./ckpts/model.pt

# Use CPU instead of GPU
python inference.py --image image.png --task parsing --device -1

# Use specific GPU
python inference.py --image image.png --task parsing --device 1
```

### Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--image` | Path to input image (required) | - |
| `--task` | Task to perform: `parsing`, `landmarks`, `headpose`, `attributes`, `age_gender_race`, `visibility` (required) | - |
| `--output-dir` | Directory to save results | `./results` |
| `--model-path` | Path to model weights (downloads if not provided) | Auto-download from HF |
| `--device` | GPU device number (-1 for CPU) | `0` |

### Getting Help

```bash
python inference.py --help
```

<p align="center" width="100%">
  <img src='docs/static/images/qualitative.png'>
</p>

## TODOs
- Release dataloaders for the datasets used.
- Release training script.

## Citation
If you find _FaceXFormer_ useful for your research, please consider citing us:

```bibtex
@article{narayan2024facexformer,
  title={FaceXFormer: A Unified Transformer for Facial Analysis},
  author={Narayan, Kartik and VS, Vibashan and Chellappa, Rama and Patel, Vishal M},
  journal={arXiv preprint arXiv:2403.12960},
  year={2024}
}
```
## Contact
If you have any questions, please create an issue on this repository or contact at knaraya4@jhu.edu
