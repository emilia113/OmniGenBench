<!-- start -->

<div align="center">
<img src="assets/omnigenbench.png" alt="OmniGenBench Logo" width="58">

<h2>OmniGenBench: A Benchmark for Omnipotent Multimodal Generation across 50+ Tasks</h2>

<a href="https://github.com/emilia113">Jiayu Wang</a><sup>1,2*</sup>&nbsp;
<a href="https://sxjyjay.github.io/">Yang Jiao</a><sup>1,2*</sup>&nbsp;
<a href="https://github.com/Yue-105">Yue Yu</a><sup>1,2</sup>&nbsp;
<a href="https://qiantianwen.github.io/">Tianwen Qian</a><sup>3</sup>&nbsp;
<br>
<a href="https://scholar.google.com/citations?user=WL5mbfEAAAAJ&hl=zh-CN">Shaoxiang Chen</a><sup>4</sup>&nbsp;
<a href="https://jingjing1.github.io/">Jingjing Chen</a><sup>1,2†</sup>&nbsp;
<a href="https://fvl.fudan.edu.cn/">Yu-Gang Jiang</a><sup>1,2</sup>

<sup>1</sup>Shanghai Key Lab of Intell. Info. Processing, School of CS, Fudan University<br>
<sup>2</sup>Shanghai Collaborative Innovation Center on Intelligent Visual Computing<br>
<sup>3</sup>School of Computer Science and Technology, East China Normal University<br>
<sup>4</sup>Minimax  

<!-- Uncomment below to show paper and huggingface badge -->
[![OmniGenBench](https://img.shields.io/badge/Paper-OmniGenBench-d32f2f.svg?logo=arXiv)](https://arxiv.org/abs/2505.18775)&#160;
[![OmniGenBench](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20-OmniGenBench-blue)](https://huggingface.co/datasets/emiliiia/OmniGenBench)


</div>
<br>

## 📢 News


- **[2025/05/27]** Our paper is released.
- **[2025/05/26]** The benchmark and evaluation code will be released soon.

<br>
<br>

## 📝 TODO

- [ ] Release leaderboard
- [ ] Release evaluation code
- [x] Release benchmark data

<br>
<br>

## 💡 Introduction

</div>
<div align="center">
  <img src="assets/class.png" width="92%">
</div>
With <b>50+ diverse sub-tasks</b>, OmniGenBench serves as a novel and comprehensive benchmark for evaluating the generality, adaptability, and reasoning capabilities of state-of-the-art generative models across both perception- and cognition-oriented generation tasks.
<br><br>
OmniGenBench ensures task diversity and difficulty by building on MegaBench, a widely acknowledged multimodal benchmark. We reverse-engineer text queries from its tasks and apply human filtering for accuracy and challenge.
<br><br>
To enable precise evaluation, we design dedicated evaluation protocols for both perception- and cognition-centric tasks. Each task is assigned a tailored evaluation criterion, and our protocol aligns closely with human judgment, ensuring both accuracy and consistency in performance assessment.


<br>
<br>

## 🚀 Submit Your Model

We’re currently running evaluations on OmniGenBench — a comprehensive benchmark covering 50+ real-world tasks designed to test the limits of general-purpose image generation models.

If you’d like to see how your model performs across a variety of challenging and diverse prompts, feel free to leave a comment in the Issues section — GitHub repo, API endpoint, Hugging Face Space, or any other inference link are all welcome!

<br>
<br>


## 🛠️ Quick Start
### 1. Download the Benchmark Data

You can download the OmniGenBench benchmark dataset from Hugging Face:
[https://huggingface.co/datasets/emiliiia/OmniGenBench](https://huggingface.co/datasets/emiliiia/OmniGenBench)

### 2. Environment Setup

Install Python 3.10 and clone the benchmark repository:

```bash
git clone https://github.com/emilia113/OmniGenBench.git
cd OmniGenBench
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Image Generation

1. Replace the `GenerationModel` class in `generated_image.py` with your own image generation model.

2. Run the script to generate images:

```bash
python generate_image.py \
    --prompt_root /path/to/benchmark/text \
    --image_root /path/to/benchmark/image \
    --log_save_dir ./log \
    --result_save_dir ./generated_images
```

### 4. Evaluation

1. Save your Google Gemini API key to a text file (e.g., `my_genai_key.txt`).

2. Run the evaluation script:

```bash
python cal_metrics.py \
    --genai_key_file /path/to/my_genai_key.txt \                            # Your Gemini API key file
    --benchmark_root /path/to/OmniGenBench/benchmark \                      # Path to benchmark data
    --generated_image_root /path/to/generated_images/GenerationModel \      # Path to generated images
    --eval_material_root /path/to/OmniGenBench/eval_material \              # Evaluation materials
    --only_instruction_following                                            # Only evaluate instruction-following consistency
```

## 🎨 Model Outputs Showcase
<div align="center">
  <img src="assets/exper.png" width="100%">
</div>
