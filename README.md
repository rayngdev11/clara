<h1 align="center">CLARA - Vision-Language AI for Chest X-ray Caption Automation</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688.svg" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Inference-vLLM-blueviolet.svg" alt="vLLM"/>
  <img src="https://img.shields.io/badge/Deployment-Triton%20Server-76B900.svg" alt="Triton"/>
  <img src="https://img.shields.io/badge/Model-Qwen2--VL--7B-orange.svg" alt="Model"/>
</p>

## Overview

This repository contains the source code and deployment infrastructure for **CLARA**, an AI system designed to automatically generate detailed clinical findings and impressions for chest X-rays in Vietnamese. 

This project was developed as a Capstone Project by students at **FPT University**. The goal is to serve as an assistive tool to reduce radiologist workload, enhance linguistic consistency, shorten processing time, and improve diagnostic accuracy.

## Key Highlights

- **Massive Medical Dataset (ViX-Ray):** Fine-tuned on a comprehensive, high-quality dataset of Vietnamese medical chest X-ray samples.
- **State-of-the-Art VLM (Qwen2-VL-7B-Instruct):** Chosen for its dynamic resolution vision encoder and superior multilingual alignment.
- **Progressive 3-Stage Training Pipeline:** 
  1. **Findings Generation:** Training the model to recognize anatomical features and pathologies.
  2. **Impressions Generation:** Summarizing findings into accurate diagnostic conclusions.
  3. **Multi-turn Chat:** Enabling interactive conversational diagnoses.
- **Optimized for High-Performance Hardware:** Trained using **DeepSpeed ZeRO-3, CPU Offload, Gradient Checkpointing, and BF16 precision** across 7x NVIDIA RTX 5090 GPUs.

## System Architecture & Deployment

The system is designed with a **Production-Ready Mindset**, featuring 3 distinct inference backends:

1. **Native PyTorch (HuggingFace):** Traditional inference, suitable for development and debug.
2. **vLLM-Accelerated Serving:** Provides high-throughput, memory-efficient serving designed tailored for large language models. Features continuous batching and PagedAttention.
3. **NVIDIA Triton Inference Server (Docker):** Enterprise-ready, highly scalable containerized framework featuring dynamic batching and optimal resource orchestration.

### Inference Pipeline
- **Frontend Context Input**: A web-based user interface or Gradio interface dynamically constructs prompt templates embedding the patient's Age and Gender.
- **FastAPI Gateway**: Orchestrates requests and passes image tensors + queries to the inference engine.
- **Two-Turn Reasoning (Chain of Thought)**: 
  - *Turn 1*: Extract detailed visual findings.
  - *Turn 2*: Consolidate extracted findings into a final diagnosis.

## Evaluation Metrics
Our models are subjected to rigorous evaluations aligning with clinical and NLP standards:
- **Lexical Quality:** Evaluated automatically using **BLEU** and **ROUGE-L** scores.
- **Clinical Factual Integrity:** Assessed via **Precision** and **Recall** of atomic medical facts.
- **Doctor Benchmark:** Qualitatively vetted by active radiologists at Military Hospital 175 on diagnostic accuracy and information completeness on a 1-5 scale. *Qwen2-VL-7B significantly outperformed smaller baseline models across all testing methodologies.*

## System Requirements

> **⚠️ Minimum Hardware Notice**  
> Running the 7 Billion parameter `Qwen2-VL-7B` model requires substantial computational resources. 
> - **GPU/VRAM:** Minimum **24GB VRAM** (e.g., RTX 3090, RTX 4090, A10) for inference.
> - **Storage:** Sufficient space to store model weights (approx. 15-20 GB depending on precision).
> - **OS:** Ubuntu/Linux or Windows Subsystem for Linux (WSL2) is strongly recommended for `vllm` and `triton` compatibility.

- **Python 3.8+**
- **Docker & Docker Compose** (for Triton Server)

## Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/rayngdev11/clara.git
cd clara
```

**2. Install Dependencies**

Since inference libraries (`vllm`, `tritonclient`, etc.) are heavily system-dependent, they must be installed based on your exact CUDA version. A general requirements environment might include:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install vllm fastapi uvicorn gradio openai pillow
```
> *(Note: A dedicated `requirements.txt` should be generated based on your target system)*

**3. Model Weights Delivery**
To run inference, you must download the specified fine-tuned weights (e.g., from the provided HuggingFace repositories outlined in the project thesis) and update the `model_path` variable in `src/app_vllm/api.py`.

### Recommended Inference (vLLM)

1. **Start the vLLM Engine**
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/qwen2-vl-7b-weights --port 8006
```

2. **Start the FastAPI Backend**
```bash
cd src/app_vllm
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

3. **Start the Frontend Interface**
You can use the built-in Gradio UI for quick testing:
```bash
# Launch a doctor-facing interactive UI
cd src/app_vllm
python gradio_doctor.py
```


## Authors
This project is brought to you by the Capstone Project team supervised by **Truong Hoang Vinh**.

## License
This project is licensed under the MIT License.
