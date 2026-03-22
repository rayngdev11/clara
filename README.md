<h1 align="center">CLARA — Vision-Language AI for Chest X‑ray Reporting</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg"/>
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red"/>
  <img src="https://img.shields.io/badge/FastAPI-API-009688"/>
  <img src="https://img.shields.io/badge/vLLM-Inference-blueviolet"/>
  <img src="https://img.shields.io/badge/Triton-Serving-76B900"/>
  <img src="https://img.shields.io/badge/Model-Qwen2--VL--7B-orange"/>
</p>

---

##  What is CLARA?

CLARA is a **production‑ready vision‑language AI system** that automatically generates structured radiology reports (Findings & Impressions) from chest X‑rays in **Vietnamese**.  
It was developed as a **Capstone Project** at **FPT University** and is designed to assist radiologists by reducing workload, improving consistency, and accelerating clinical workflows.

---

##  Key Results

| Metric | Value |
|--------|-------|
| **BLEU Score** | 89.2 |
| **Blind‑Pass Rate (Doctor Evaluation)** | 74.5% |
| **Training Data** | 137,930 chest X‑ray examinations |

*The fine‑tuned Qwen2‑VL‑7B significantly outperformed smaller baseline models in both lexical quality and clinical acceptance.*

---

##  Technical Deep Dive

### Model & Training
- **Base model:** [Qwen2‑VL‑7B‑Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) – chosen for its dynamic vision encoder and strong multilingual alignment.
- **Training strategy:** 3‑stage progressive pipeline (Findings → Impressions → Multi‑turn chat) using **DeepSpeed ZeRO‑3**, **CPU offload**, gradient checkpointing, and **BF16** precision.
- **Infrastructure:** Distributed training across **7 × NVIDIA RTX 5090** GPUs.

### System Architecture
- **Multi‑stage reasoning:**  
  1. Image encoding  
  2. Detailed findings generation  
  3. Diagnostic impression generation  
  4. Optional multi‑turn refinement
- **Inference backends:**  
  - **vLLM** – high‑throughput serving with continuous batching  
  - **NVIDIA Triton** – enterprise‑grade, containerized deployment  
  - Native PyTorch – for development and debugging
- **API & Frontend:** FastAPI gateway + Gradio UI for easy clinician interaction.

---

##  Skills Demonstrated

| Category | Technologies / Methods |
|----------|------------------------|
| **AI/ML** | Vision‑Language Models (VLM), Large Language Models (LLM), Fine‑tuning (LoRA), DeepSpeed, BF16, Chain‑of‑Thought |
| **Backend & API** | FastAPI, RESTful API design, asynchronous processing |
| **Inference Optimization** | vLLM (PagedAttention, continuous batching), NVIDIA Triton Inference Server, Docker containerization |
| **System Integration** | Modular pipeline design, scalable architecture, production‑grade deployment |
| **Evaluation** | BLEU, ROUGE‑L, precision/recall for clinical facts, blind evaluation by radiologists |

---

##  Hardware & Requirements

- **GPU:** Minimum 24GB VRAM (e.g., RTX 3090, RTX 4090, A10) for inference; training required multi‑GPU setup.
- **Storage:** ~15–20 GB for model weights.
- **OS:** Ubuntu/Linux or WSL2 recommended for vLLM and Triton compatibility.
- **Python 3.8+**, Docker, and Docker Compose.

---
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
