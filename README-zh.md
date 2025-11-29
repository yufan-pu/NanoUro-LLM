<div align="center">

# 🧬 NanoUro-LLM
### 泌尿专科微调推理大语言模型

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Unsloth-Powered-green.svg)](https://github.com/unslothai/unsloth)
[![Model](https://img.shields.io/badge/Model-DeepSeek%2FPyTorch-red.svg)](https://pytorch.org/)

**NanoUro-LLM** 是一个专为**泌尿科 (Urology)** 临床场景设计的轻量级大语言模型微调框架。
基于 [Unsloth](https://github.com/unslothai/unsloth) 加速引擎与 [RJUA-QADatasets](http://data.openkg.cn/dataset/rjua-qadatasets)，致力于将 DeepSeek-R1、Gemma 3 等通用模型转化为具备一定泌尿临床知识的专科 AI 助手。

[English](README-en.md) • [中文](README-zh.md) • [项目特点](#-核心特性) • [安装部署](#-安装部署) • [微调指南](#-微调工作流) • [演示案例](#-推理演示) • [局限性](#-局限性)

</div>

---

## 🏥 项目背景

通用大语言模型（General LLMs）在处理特定医学专科问题时，常面临**专业知识匮乏**与**推理成本过高**的双重挑战。

NanoUro-LLM 针对**仁济医院泌尿科问答数据集**构建了一套完整的 `ETL -> LoRA Fine-tuning -> Inference` 流水线。低成本训练垂直领域医学推理模型的试验方案。

### ✨ 核心特性

* **⚡ 微调加速**：基于 `Unsloth` 框架，显存占用降低，训练速度提升。
* **🧠 医学思维链**：引入 "Medical CoT" 数据结构，训练模型在回答前先进行病理分析与鉴别诊断。
* **🧹 智能清洗**：内置针对临床文本的正则预处理引擎，自动去除学术引用（如 `[12-15]`）与格式噪声。
* **🔄 模块化架构**：统一接口支持多模态家族：
    * `DeepSeek-R1-Distill` (Llama/Qwen) - 推荐 🔥
    * `Gemma-3` / `Qwen-3`
* **📦 灵活部署**：原生支持 ModelScope、HuggingFace 及本地加载。

---

## 📊 数据集与模型支持

### 训练数据
本模型基于 **RJUA-QADatasets**（蚂蚁集团-仁济医院泌尿科问答推理数据集）构建。
- **结构**：`{问题, 上下文(医学知识), 答案, 疾病, 建议}`
- **预处理**：针对非结构化医学文本进行标准化清洗与去噪。

### 支持模型矩阵
框架目前支持以下经过优化的模型架构，同时兼容其他在[huggingface](https://huggingface.co/unsloth) /[modelscope](https://www.modelscope.cn/organization/unsloth?tab=model)中的`unsloth/[models]`模型：

| 模型系列 | 基础架构 | 参数 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **DeepSeek-R1** | Distill-Llama | 8B | 较优逻辑推理能力，适用于需要一定准确性的医学问答和辅助诊断场景 |
| **DeepSeek-R1** | Distill-Qwen | 1.5B | 较优推理速度，适合资源受限的问答 |
| **Gemma 3** | Google Gemma 3 | 4B | 均衡，响应质量一般 |
| **Qwen 3** | Qwen 3 | 0.6B | 小参数量，移动端/边缘设备原型验证，生成质量有限 |
| **Gemma 3** |Google Gemma 3  | 270M | 最小参数量，质量较低，仅建议用于演示或非关键任务|

---

## 🛠️ 安装部署

推荐使用 NVIDIA GPU 环境（显存 ≥ 16GB 推荐，支持 A100/4090D）。

### 1. 克隆项目
```bash
git clone https://github.com/yufan-pu/NanoUro-LLM.git
cd NanoUro-LLM
```

### 2. 环境配置 (推荐 Mamba/Conda)
Anaconda 在处理部分依赖时可能较慢，推荐使用 Mamba。

```bash
# 创建环境 (Python 3.12)
mamba create -n nanouroLLM python=3.12 ipykernel jupyterlab -y
mamba activate nanouroLLM

# 注册 Jupyter Kernel
python -m ipykernel install --user --name nanouroLLM --display-name "Python 3.12 (nanouroLLM)"
```

### 3. 安装 PyTorch 
请根据你的 CUDA 版本安装：

```bash
# 示例：安装兼容 CUDA 12.8 的 PyTorch (请根据 pytorch.org 实际情况调整)
# 确定GPU和cuda支持版本，CUDA Version: 12.8 → 对应安装cu128版本Torch
nvidia-smi 
# 安装torch,参考官方示例
https://pytorch.org/get-started/previous-versions/ 
# 例：安装torch 2.9.0+cu128
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```
###  4.安装 Unsloth 及其他依赖
* 在IDE打开项目文件`NanoUrollm.ipynb`  
  然后运行`install required libraries`安装

---

## 🚀 微调工作流
* 打开`NanoUrollm.ipynb`运行，项目通过 `Data curation`,`MedicalFineTuningFramework` 类封装了数据清洗及LoRA微调等流程。

### 第一步：数据清洗
RJUA的原始JSON数据包含学术引用和格式噪声，运行`Data curation`模块
* 输入`RJUA_train.json`
* 输出`RJUA_train_cleaned.json`文件

### 第二步：启动微调
配置 LoRA 参数一键运行开始训练，框架自动处理分词器和对话模板对齐。

```python

# 1. 初始化框架与模型 (支持 ModelScope/HF)
framework = MedicalFineTuningFramework()
framework.setup_model(
    model_key="deepseek-8b",
    download_source="modelscope"
)

# 2. 配置 LoRA 适配器 (根据显存调整 r/alpha/dropout 值)
framework.setup_lora(r=16, alpha=16, dropout=0.05)

# 3. 加载数据并训练保存
framework.setup_data("data/RJUA_train_cleaned.json")
stats = framework.train(
    output_dir="outputs/deepseek-8b_lora_v1",
    max_steps=300, # samples size=1705 /batchsize=4/gradient_accumulation_steps=2 → 214+ steps(1 epoch)
    batch_size=4
)
```

### 第三步：加载与推理
使用微调好的 LoRA adapter 进行推理，建议使用绝对路径,

```python
from inference import ModelLoader

# 模型路径配置
BASE_MODEL_DIR = "./model_cache/unsloth/DeepSeek-R1-Distill-Llama-8B"
LORA_OUTPUT_DIR = "./outputs/deepseek-8b_lora"


# 创建自定义推理配置
custom_config = InferenceConfig(
    temperature=0.6,
    top_p=0.95,
    top_k=64,
    repetition_penalty=1.15,
    max_new_tokens=2048,
    enable_thinking=True
)

# 系统提示词
system_instruction = "你是一个泌尿科医学专家，请根据患者描述和医学知识，详细推理后给出专业诊断和治疗建议。"
# 用户提示词
user_input = "75岁男性，最近尿频尿急，请问可能是什么原因？"

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

## 📝 推理演示

*来自基于此数据集微调的DeepSeek-8B模型的实际输出：*

> **用户（患者）：**
> "你好，我最近发热腰痛不适，应该怎么办？"

> **NanoUro-LLM：**
> **（思维过程）**
> *...患者主诉尿频尿急一周...CT显示双侧肾结石，右肾下极结石...输尿管结石...*
> *分析：症状提示尿路感染合并结石引起的梗阻。*
>
> **（回答）**
> 根据您的症状和检查结果，您的诊断为**泌尿道炎症**（尿道口酸痛、尿频尿急）合并**双侧输尿管结石**。
>
> **治疗建议：**
>
> 1.  **抗感染**：泌尿道炎症需要抗生素治疗。
> 2.  **手术介入**：泌尿系结石伴有积水/梗阻，建议进行手术治疗（如输尿管镜碎石术）。
> 3.  **对症治疗**：针对肝小囊肿建议后续随访或微波消融。
---

## ⚠️ 局限性

本框架及模型仅供**医学 AI 研究与技术验证**使用。

| 局限性 | 说明 |
| :--- | :--- |
| **🧠 模型幻觉** | 大型语言模型本质上是一种概率生成模型，**可能生成看似合理但事实上不准确、不完整或完全错误（即"幻觉"）的医学信息**。这包括但不限于：编造不存在的症状、推荐错误的治疗方案、误诊病情等。**不可将其输出视为最终的医学诊断或治疗依据。** |
| **📊 未经系统性临床评估** | 模型**尚未经过严格的、大规模的临床前瞻性试验或由权威医学专家委员会的全面评估**。其有效性和安全性仍未得到充分验证。模型表现基于有限的测试数据，在真实世界的复杂医疗环境中可能表现不佳。 |
| **🔬 知识截止与覆盖范围** | 模型的训练数据存在知识截止日期，**无法涵盖最新的医学研究、临床指南、药品信息或疾病发现**。同时，其知识库可能无法覆盖所有罕见病、复杂病症或特定人群的医学情况。 |
| **🤖 小模型的固有缺陷** | 轻量级模型（如 `qwen3-0.6b` 和 `gemma3-270m`）**表达能力有限**。更可能出现逻辑混乱、事实错误、前后矛盾或"胡言乱语"的情况，**在医学领域的实用性低，仅建议用于技术演示或非关键场景的探索**。 |
| **⚖️ 泛化能力与偏见** | 模型的性能严重依赖于其训练数据。如果训练数据存在分布偏差（例如，某些人群或疾病的数据不足），模型在相应领域的表现会下降，甚至可能放大数据中存在的社会偏见或医学实践差异。 |

---

## 📜 许可证

* **代码**：Apache 2.0 License
* **数据集**：遵循 RJUA / OpenKG 原始协议
* **模型权重**：遵循基础模型（DeepSeek/Google/Qwen）的使用协议

<div align="center">
<br/>
<sub>Made with 奶牛猫 | Powered by <a href="https://github.com/unslothai/unsloth">Unsloth</a> 和医学热情</sub></sub>
</div>
