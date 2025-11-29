<div align="center">

# 🧬 NanoUro-LLM
### Urology-Specialized Fine-Tuned Large Language Model

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Unsloth-Powered-green.svg)](https://github.com/unslothai/unsloth)
[![Model](https://img.shields.io/badge/Model-DeepSeek%2FPyTorch-red.svg)](https://pytorch.org/)

**NanoUro-LLM** is a lightweight large language model fine-tuning framework specifically designed for **urology** clinical scenarios.
Based on the [Unsloth](https://github.com/unslothai/unsloth) acceleration engine and [RJUA-QADatasets](http://data.openkg.cn/dataset/rjua-qadatasets), it aims to transform general models like DeepSeek-R1 and Gemma 3 into specialized AI assistants with urology clinical knowledge.

[English](README-en.md) • [中文](README-zh.md) • [Features](#-core-features) • [Installation](#️-environment--installation) • [Fine-tuning Guide](#-fine-tuning-workflow) • [Demo](#-inference-demonstration) • [Disclaimer](#️-limitations--disclaimer)

</div>

---

## 🏥 Project Background

General large language models (LLMs) often face dual challenges of **lack of specialized knowledge** and **high inference costs** when dealing with specific medical specialty problems.

NanoUro-LLM builds a complete `ETL -> LoRA Fine-tuning -> Inference` pipeline targeting the **Renji Hospital Urology Q&A Dataset**, providing an experimental solution for low-cost training of vertical domain medical reasoning models.

### ✨ Core Features

* **⚡ Fine-tuning Acceleration**: Based on the `Unsloth` framework, reduces GPU memory usage and increases training speed.
* **🧠 Medical Chain-of-Thought**: Introduces "Medical CoT" data structure, training models to perform pathological analysis and differential diagnosis before answering.
* **🧹 Intelligent Cleaning**: Built-in regex preprocessing engine for clinical text, automatically removing academic citations (like `[12-15]`) and formatting noise.
* **🔄 Modular Architecture**: Unified interface supports multiple model families:
    * `DeepSeek-R1-Distill` (Llama/Qwen) - Recommended 🔥
    * `Gemma-3` / `Qwen-3`
* **📦 Flexible Deployment**: Native support for ModelScope, HuggingFace, and local loading.

---

## 📊 Dataset & Model Support

### Training Data
This model is built on **RJUA-QADatasets** (Ant Group-Renji Hospital Urology Q&A Reasoning Dataset).
- **Structure**: `{question, context(medical knowledge), answer, disease, advice}`
- **Preprocessing**: Standardized cleaning and denoising for unstructured medical text.

### Supported Model Matrix
The framework currently supports the following optimized model architectures, while also being compatible with other `unsloth/[models]` from [huggingface](https://huggingface.co/unsloth) / [modelscope](https://www.modelscope.cn/organization/unsloth?tab=model):

| Model Family | Base Architecture | Parameters | Use Case |
| :--- | :--- | :--- | :--- |
| **DeepSeek-R1** | Distill-Llama | 8B | Better logical reasoning capability, suitable for medical Q&A and auxiliary diagnosis scenarios requiring certain accuracy |
| **DeepSeek-R1** | Distill-Qwen | 1.5B | Better inference speed, suitable for resource-constrained Q&A |
| **Gemma 3** | Google Gemma 3 | 4B | Balanced, average response quality |
| **Qwen 3** | Qwen 3 | 0.6B | Small parameter count, mobile/edge device prototyping, limited generation quality |
| **Gemma 3** | Google Gemma 3 | 270M | Minimum parameter count, lower quality, recommended only for demos or non-critical tasks |

---

## 🛠️ Environment & Installation

Recommended to use NVIDIA GPU environment (VRAM ≥ 16GB recommended, supports A100/4090D).

### 1. Clone Project
```bash
git clone [https://github.com/yufan-pu/NanoUro-LLM.git](https://github.com/yufan-pu/NanoUro-LLM.git)
cd NanoUro-LLM
```

### 2. Environment Setup (Recommended Mamba/Conda)

Anaconda might be slower with some dependencies, Mamba is recommended.

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
# Example: Install PyTorch compatible with CUDA 12.8 (adjust according to actual situation on pytorch.org)
# Determine GPU and CUDA support version, CUDA Version: 12.8 → Install cu128 version Torch
nvidia-smi 
# Install torch, refer to official example
[https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/) 
# Example: Install torch 2.9.0+cu128
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
```

### 4. Install Unsloth & Other Dependencies

* Open the project file `NanoUrollm.ipynb` in IDE  
  Then run `install required libraries` to install

---

## 🚀 Fine-tuning Workflow

* Open `NanoUrollm.ipynb` and run. The project encapsulates data cleaning and LoRA fine-tuning processes through `Data curation` and `MedicalFineTuningFramework` classes.

### Step 1: Data Cleaning

RJUA's original JSON data contains academic citations and formatting noise. Run the `Data curation` module.

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

# 3. Load data and train & save
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
# Configure model paths
BASE_MODEL_DIR = "./model_cache/unsloth/DeepSeek-R1-Distill-Llama-8B"
LORA_OUTPUT_DIR = "./outputs/deepseek-8b_lora"

# Create DeepSeek specific configuration
config = ModelConfig.create_deepseek_config()

# Initialize loader
loader = UnifiedModelLoader(config)

try:
    # Load model
    loader.load_local_lora(BASE_MODEL_DIR, LORA_OUTPUT_DIR)
    
    # System prompt
    system_prompt = "You are a urology medical expert. Please provide professional diagnosis and treatment recommendations based on patient descriptions and medical knowledge, with detailed reasoning."
    
    # Test inference
    user_input = "Doctor, my 70-year-old father urinates frequently at night, about 3-4 times, and also frequently during the day. He has been taking doxazosin and finasteride regularly. Because he had a cardiac stent placed 3 years ago, he has been taking aspirin and plavix long-term. This time he went to the hospital because he had blood in his urine for 2 days. Examination showed a bladder blood clot of 7*7cm. Blood tests: white blood cells: 5.21, hemoglobin: 151, serum creatinine: 71, PT: 11.3, INR: 0.96. What is my father's condition and how should it be handled?"
    
    print("🧪 Testing DeepSeek inference (with thinking chain enabled):")
    response = loader.chat(
        user_input,
        system_prompt=system_prompt,
        enable_thinking=True,
        max_new_tokens=4096
    )
    print(f"🤖 NanoUro-LLM: {response}")
```

---

## 📝 Inference Demonstration

*Actual output from DeepSeek-8B model fine-tuned on this dataset:*

> **User (Patient):**
> "Doctor, my 70-year-old father urinates frequently at night, about 3-4 times, and also frequently during the day. He has been taking doxazosin and finasteride regularly. Because he had a cardiac stent placed 3 years ago, he has been taking aspirin and plavix long-term. This time he went to the hospital because he had blood in his urine for 2 days. Examination showed a bladder blood clot of 7*7cm. Blood tests: white blood cells: 5.21, hemoglobin: 151, serum creatinine: 71, PT: 11.3, INR: 0.96. What is my father's condition and how should it be handled?"

> **NanoUro-LLM:**  
> **(Thinking Process)** *First, considering your father's advanced age and history of heart disease (coronary artery bypass surgery), this will affect his overall health status. Blood in urine may indicate bladder or urethral infection, or more serious issues like kidney cancer, acute pyelonephritis, or other urinary system diseases. Combined with the bladder blood clot size of 7×7 cm, slightly elevated white blood cells (5.21), slightly high serum creatinine (71), and normal coagulation function (PT: 11.3, INR: 0.96), initial possible causes include:... In summary, your father's condition requires comprehensive consideration of multiple factors including bladder lesions, kidney function, heart disease history, and personal metabolic status. Precise laboratory tests, imaging assessments, and clinical trials are needed to clarify the diagnosis, and then develop appropriate treatment plans...*
>
> **(Answer)**
> **Clinical Conclusion & Treatment Decision:**
> Given your father is 70 years old with a history of heart disease, facing clinical challenges with hematuria, multiple possibilities including bladder infection, pyelonephritis, kidney stones need to be considered. Necessary imaging and laboratory tests are recommended to clarify the cause, and corresponding treatment measures should be taken based on specific diagnostic results. Meanwhile, close monitoring of heart and kidney function changes is needed to ensure safety and effectiveness during treatment.
>
> **Final Recommendations:**
> 1. **Perform urine culture and bladder nucleic acid testing**: Determine if there's bladder bacterial infection to guide treatment.
> 2. **Perform abdominal ultrasound or CT scan**: Assess bladder and renal pelvis morphology, exclude potential issues like renal pelvis stones.
> 3. **Regularly monitor kidney function and blood indicators**: Ensure treatment doesn't damage kidney function and heart health.
> 4. **Adjust treatment plan based on diagnosis**: If condition is stable, normal lifestyle can be resumed while controlling heart disease.

---

## ⚠️ Limitations & Disclaimer

This framework and model are for **medical AI research and technical validation only**.

| Limitation | Description |
| :--- | :--- |
| **🧠 Model Hallucination** | Large language models are inherently probabilistic generative models that **may generate medically inaccurate, incomplete, or completely false information (i.e., "hallucinations")** that appear plausible. This includes but is not limited to: fabricating non-existent symptoms, recommending incorrect treatments, misdiagnosing conditions. **Do not treat their output as final medical diagnosis or treatment basis.** |
| **📊 Not Systematically Clinically Evaluated** | The model **has not undergone rigorous, large-scale clinical prospective trials or comprehensive evaluation by authoritative medical expert committees**. Its effectiveness and safety remain insufficiently validated. Model performance is based on limited test data and may perform poorly in real-world complex medical environments. |
| **🔬 Knowledge Cutoff & Coverage** | The model's training data has a knowledge cutoff date and **cannot cover the latest medical research, clinical guidelines, drug information, or disease discoveries**. Additionally, its knowledge base may not cover all rare diseases, complex conditions, or medical situations in specific populations. |
| **🤖 Inherent Limitations of Small Models** | Lightweight models (like `qwen3-0.6b` and `gemma3-270m`) have **limited expressive capabilities**. They are more prone to logical confusion, factual errors, internal contradictions, or "nonsensical" output. **Their practicality in the medical field is low, recommended only for technical demonstrations or exploration of non-critical scenarios.** |
| **⚖️ Generalization Ability & Bias** | Model performance heavily depends on its training data. If training data has distribution biases (e.g., insufficient data for certain populations or diseases), model performance in corresponding areas will degrade, and it may even amplify social biases or medical practice variations present in the data. |

---

## 📜 License

* **Code**: Apache 2.0 License
* **Dataset**: Follows original RJUA / OpenKG agreements
* **Model Weights**: Follow usage agreements of base models (DeepSeek/Google/Qwen)

<div align="center">
<br/>
<sub>Made with a Tuxedo Cat | Powered by <a href="https://github.com/unslothai/unsloth">Unsloth</a> and medical passion</sub>
</div>
