# BigData-for-DynamicID

### Big Data Analysis Course Project – Thuy Loi University

---

## 📘 Project Overview

**BigData-for-DynamicID** is a **final course project** for the subject **Big Data Analysis** at **Thuy Loi University**, conducted by a group of undergraduate students.
The project focuses on **large-scale data processing and analysis** in the context of **personalized image generation**, inspired by the research work **DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability (ICCV 2025)**.

This repository is **not an official reimplementation** of the original paper. Instead, it serves as an **academic study and experimental implementation**, aiming to understand how big data techniques are applied in modern generative AI systems.

---

## 🎯 Objectives

- Study and analyze the **DynamicID framework** from a big data perspective
- Explore **large-scale facial image datasets**, preprocessing pipelines, and data organization
- Implement and experiment with **identity-aware image generation workflows**
- Demonstrate the role of **data preprocessing, scalability, and reproducibility** in big data systems

---

## 🧠 Background: DynamicID

The original **DynamicID** framework proposes a tuning-free approach for both single-ID and multi-ID personalized image generation, enabling:

- Identity preservation using reference images
- Multi-identity image synthesis
- Flexible and independent facial expression editing via text prompts

Key components introduced in the original work include:

- **Semantic-Activated Attention (SAA)**
- **Identity-Motion Reconfigurator (IMR)**
- A large-scale facial dataset (VariFace-10k)

This project **leverages the ideas and structure** of DynamicID for educational and analytical purposes.

---

## 🧩 Project Scope

In this course project, we focus on:

- Handling and preprocessing **large image datasets**
- Analyzing identity-related features in facial data
- Running training and inference experiments in **Kaggle Notebook environments**
- Evaluating results from a **big data processing viewpoint**, rather than proposing new algorithms

---

## ⚙️ Technologies & Tools

- Python
- PyTorch
- Stable Diffusion (v1.5)
- Kaggle Notebook (GPU-based experimentation)
- Image preprocessing & data pipeline techniques

---

## 📂 Repository Structure

```
bigdata-for-dynamicID/
├── notebooks/        # Data preprocessing & experiments
├── data/             # Dataset samples / structure description
├── results/          # Generated outputs
├── reports/          # Course report & analysis
└── README.md
```

---

## 📌 Academic Disclaimer

This repository is created **solely for educational purposes** as part of a university course project.
All core ideas, model designs, and methodological contributions belong to the **original DynamicID authors**.

---

## 🙏 Acknowledgement

This project is inspired by and based on the official DynamicID research and implementation:

- **Paper**: _DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability_ (ICCV 2025)
- **Official Repository**: [https://github.com/ByteCat-bot/DynamicID](https://github.com/ByteCat-bot/DynamicID)
- **Authors**: Xirui Hu et al.

We sincerely thank the authors for making their work publicly available.
