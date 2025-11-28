<div align="center">

# 🧬 NanoUro-LLM
### Fine-tuned Large Language Model for Urology Specialization

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Unsloth-Powered-green.svg)](https://github.com/unslothai/unsloth)
[![Model](https://img.shields.io/badge/Model-DeepSeek%2FPyTorch-red.svg)](https://pytorch.org/)

**NanoUro-LLM** is a lightweight large language model fine-tuning framework specifically designed for **Urology** clinical scenarios.
Built on the [Unsloth](https://github.com/unslothai/unsloth) acceleration engine and [RJUA-QADatasets](http://data.openkg.cn/dataset/rjua-qadatasets), it aims to transform general models like DeepSeek-R1 and Gemma 3 into specialized AI assistants with urological clinical knowledge.

[English](README-en.md) • [中文](README-zh.md) • [Features](#-core-features) • [Installation](#-environment--installation) • [Fine-tuning](#-fine-tuning-workflow) • [Demo](#-inference-demo) • [Disclaimer](#-limitations--disclaimer)

</div>

---

## 🏥 Project Background

General Large Language Models (LLMs) often face dual challenges of **lack of specialized knowledge** and **high inference costs** when dealing with specific medical specialty problems.

NanoUro-LLM constructs a complete `ETL -> LoRA Fine-tuning -> Inference` pipeline for the **Renji Hospital Urology Q&A Dataset**, providing an experimental solution for low-cost training of vertical domain medical reasoning models.

### ✨ Core Features

* **⚡ Accelerated Fine-tuning**: Based on the `Unsloth` framework, reducing VRAM usage and improving training speed.
* **🧠 Medical Chain-of-Thought**: Introduces "Medical CoT" data structure, training models to perform pathological analysis and differential diagnosis before answering.
* **🧹 Intelligent Cleaning**: Built-in regex preprocessing engine for clinical text, automatically removing academic citations (like `[12-15]`) and format noise.
* **🔄 Modular Architecture**: Unified interface supports multiple model families:
    * `DeepSeek-R1-Distill` (Llama/Qwen) - Recommended 🔥
    * `Gemma-3` / `Qwen-3`
* **📦 Flexible Deployment**: Native support for ModelScope, HuggingFace, and local loading.

---

## 📊 Dataset & Model Support

### Training Data
This model is built on **RJUA-QADatasets** (Ant Group - Renji Hospital Urology Q&A Reasoning Dataset).
- **Structure**: `{question, context(medical knowledge), answer, disease, advice}`
- **Preprocessing**: Standardized cleaning and denoising for unstructured medical text.

### Supported Model Matrix
The framework currently supports the following optimized model architectures, while also being compatible with other `unsloth/[models]` from [huggingface](https://huggingface.co/unsloth) / [modelscope](https://www.modelscope.cn/organization/unsloth?tab=model):

| Model Family | Base Architecture | Parameters | Use Case |
| :--- | :--- | :--- | :--- |
| **DeepSeek-R1** | Distill-Llama | 8B | Better logical reasoning capability, suitable for medical Q&A and diagnostic assistance requiring accuracy |
| **DeepSeek-R1** | Distill-Qwen | 1.5B | Better inference speed, suitable for resource-constrained Q&A |
| **Gemma 3** | Google Gemma 3 | 4B | Balanced, average response quality |
| **Qwen 3** | Qwen 3 | 0.6B | Small parameter count, mobile/edge device prototyping, limited generation quality |
| **Gemma 3** | Google Gemma 3 | 270M | Minimum parameter count, lower quality, recommended for demos or non-critical tasks only |

---

## 🛠️ Environment & Installation

Recommended to use NVIDIA GPU environment (VRAM ≥ 16GB recommended, supports A100/4090D).

### 1. Clone Project
```bash
git clone https://github.com/yufan-pu/NanoUro-LLM.git
cd NanoUro-LLM
```

### 2. Environment Setup (Recommended Mamba/Conda)
Anaconda can be slow with some dependencies; Mamba is recommended.

```bash
# Create environment (Python 3.12)
mamba create -n nanouroLLM python=3.12 ipykernel jupyterlab -y
mamba activate nanouroLLM

# Register Jupyter Kernel
python -m ipykernel install --user --name nanouroLLM --display-name "Python 3.12 (nanouroLLM)"
```

### 3. Install PyTorch
Please install according to your CUDA version:

```bash
# Example: Install PyTorch compatible with CUDA 12.8 (adjust according to actual situation from pytorch.org)
# Determine GPU and CUDA support version, CUDA Version: 12.8 → Install cu128 version Torch
nvidia-smi 
# Install torch, refer to official example
https://pytorch.org/get-started/previous-versions/ 
# Example: Install torch 2.9.0+cu128
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install Unsloth & Other Dependencies
* Open the project file `NanoUrollm.ipynb` in your IDE
* Then run `install required libraries` to install dependencies

---

## 🚀 Fine-tuning Workflow
* Open `NanoUrollm.ipynb` and run. The project encapsulates data cleaning and LoRA fine-tuning processes through `Data curation` and `MedicalFineTuningFramework` classes.

### Step 1: Data Cleaning
The original JSON data from RJUA contains academic citations and format noise. Run the `Data curation` module.
* Input: `RJUA_train.json`
* Output: `RJUA_train_cleaned.json` file

### Step 2: Start Fine-tuning
Configure LoRA parameters and run with one click to start training. The framework automatically handles tokenizer and conversation template alignment.

```python

# 1. Initialize framework & model (supports ModelScope/HF)
framework = MedicalFineTuningFramework()
framework.setup_model(
    model_key="deepseek-8b",
    download_source="modelscope"
)

# 2. Configure LoRA adapter (adjust r/alpha/dropout values based on VRAM)
framework.setup_lora(r=16, alpha=16, dropout=0.05)

# 3. Load data and train/save
framework.setup_data("data/RJUA_train_cleaned.json")
stats = framework.train(
    output_dir="outputs/deepseek-8b_lora_v1",
    max_steps=300, # samples size=1705 /batchsize=4/gradient_accumulation_steps=2 → 214+ steps(1 epoch)
    batch_size=4
)
```

### Step 3: Loading & Inference
Use the fine-tuned LoRA adapter for inference. Absolute paths are recommended.

```python
from inference import ModelLoader

# Model path configuration
BASE_MODEL_DIR = "./model_cache/unsloth/DeepSeek-R1-Distill-Llama-8B"
LORA_OUTPUT_DIR = "./outputs/deepseek-8b_lora"


# Create custom inference configuration
custom_config = InferenceConfig(
    temperature=0.6,
    top_p=0.95,
    top_k=64,
    repetition_penalty=1.15,
    max_new_tokens=2048,
    enable_thinking=True
)

# System prompt
system_instruction = "You are a urology medical expert. Please provide professional diagnosis and treatment recommendations based on patient descriptions and medical knowledge, with detailed reasoning."
# User prompt
user_input = "75-year-old male, recently experiencing frequent urination and urgency. What could be the possible reasons?"

response = loader.chat(
    user_input, 
    system_prompt=system_instruction,
    temperature=0.6,
    max_new_tokens=2048,
    enable_thinking=True
)

print(response)
```

---

## 📝 Inference Demo

*Actual output from DeepSeek-8B model fine-tuned on this dataset:*

> **User (Patient):**
> "Hello, I've been experiencing fever and lower back pain recently. What should I do?"

> **NanoUro-LLM:**
> **(Thinking Process)**
> *...Patient complains of frequent urination and urgency for one week...CT shows bilateral kidney stones, right lower pole stone...ureteral stones...*
> *Analysis: Symptoms suggest urinary tract infection combined with obstruction caused by stones.*
>
> **(Answer)**
> Based on your symptoms and examination results, your diagnosis is **urinary tract inflammation** (urethral orifice soreness, frequent urination and urgency) combined with **bilateral ureteral stones**.
>
> **Treatment Recommendations:**
>
> 1.  **Anti-infection**: Urinary tract inflammation requires antibiotic treatment.
> 2.  **Surgical intervention**: Urinary stones with hydronephrosis/obstruction recommend surgical treatment (such as ureteroscopic lithotripsy).
> 3.  **Symptomatic treatment**: For small hepatic cysts, recommend follow-up observation or microwave ablation.
---

## ⚠️ Limitations

This framework and model are for **Medical AI Research and Technical Validation** only.

| Limitation | Description |
| :--- | :--- |
| **🧠 Model Hallucination** | Large language models are inherently probabilistic generative models that **may generate medically inaccurate, incomplete, or entirely false information (i.e., "hallucinations") that appear plausible**. This includes but is not limited to: fabricating non-existent symptoms, recommending incorrect treatments, misdiagnosing conditions. **Do not treat their output as definitive medical diagnosis or treatment guidance.** |
| **📊 Not Systematically Clinically Evaluated** | The model **has not undergone rigorous, large-scale clinical prospective trials or comprehensive evaluation by authoritative medical expert committees**. Its effectiveness and safety remain insufficiently validated. Model performance is based on limited test data and may perform poorly in real-world complex medical environments. |
| **🔬 Knowledge Cutoff & Coverage** | The model's training data has a knowledge cutoff date and **cannot cover the latest medical research, clinical guidelines, drug information, or disease discoveries**. Additionally, its knowledge base may not cover all rare diseases, complex conditions, or medical situations for specific populations. |
| **🤖 Inherent Limitations of Small Models** | Lightweight models (like `qwen3-0.6b` and `gemma3-270m`) have **limited expressive capability**. They are more prone to logical confusion, factual errors, internal contradictions, or "nonsensical" output. **Their practicality in the medical field is low, recommended only for technical demonstrations or exploration in non-critical scenarios.** |
| **⚖️ Generalization Ability & Bias** | Model performance heavily depends on its training data. If training data has distribution biases (e.g., insufficient data for certain populations or diseases), model performance in corresponding areas will decline, potentially amplifying social biases or medical practice variations present in the data. |

---

## 📜 License

* **Code**: Apache 2.0 License
* **Dataset**: Follows original RJUA / OpenKG agreements
* **Model Weights**: Follow usage agreements of base models (DeepSeek/Google/Qwen)

<div align="center">
<br/>
<sub>Made with 奶牛猫 | Powered by <a href="https://github.com/unslothai/unsloth">Unsloth</a> and Medical Passion.</sub>
</div>