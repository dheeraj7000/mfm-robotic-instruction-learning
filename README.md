# 🤖 MFM-Robotic-Instruction-Learning
**Memory-Efficient Multimodal Fusion Model for Vision–Language Robotic Reasoning in AI2-THOR**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)
![GPU](https://img.shields.io/badge/GPU-GTX1650%204GB-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🧠 Overview
This repository contains the implementation of a **Multimodal Fusion Model (MFM)** designed for **memory-efficient robotic instruction learning** inside the **AI2-THOR environment**.  
The model jointly processes **visual frames** and **textual prompts** to generate:
- Next-step **robot actions**
- Contextual **VQA answers**
- Natural **language descriptions**
- Continuous **task progress estimates**

All experiments were conducted on a **single NVIDIA GeForce GTX 1650 (4 GB VRAM)**, demonstrating that full multimodal fine-tuning and inference are possible on consumer hardware without quantization or checkpointing.

---

## ⚙️ Key Features
- 🧩 **Gated Cross-Attention Fusion** — Aligns image and text features efficiently.  
- 🔋 **Low-Memory Footprint** — <4 GB VRAM even during fine-tuning.  
- 🤖 **AI2-THOR Compatible** — Realistic robotic scenes with visual grounding.  
- 💬 **Multitask Learning** — Supports Action, VQA, Description, and Progress modes.  
- 🧠 **Synthetic Data Generation** — Uses Mistral via LiteLLM for dataset creation.

---

## 🧩 Model Architecture
