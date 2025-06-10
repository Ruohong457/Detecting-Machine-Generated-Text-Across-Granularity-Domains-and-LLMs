# Detecting Machine-Generated Text Across Granularity, Domains, and LLMs

This repository contains the code and (partial) data used in the master's thesis project completed at Tilburg University for the MSc in Data Science & Society.

## 📘 Project Overview

This project investigates the generalization capabilities of machine-generated text detectors under realistic distribution shifts. Specifically, it evaluates how detection models perform when:

- The **text granularity** varies (e.g., sentence, paragraph, document)
- The **content domain** shifts (e.g., academic, social media, fiction)
- The **source LLM** differs (e.g., ChatGPT, GPT-4.5, DeepSeek R1)

A multilingual approach was adopted, covering both English and Chinese datasets. A new benchmark, **AGT-2025**, is introduced to test robustness against unseen LLMs.

## ⚙️ Methods

Four transformer-based classifiers were fine-tuned and tested:

- `roberta-base` (English)
- `chinese-roberta-wwm-ext` (Chinese)
- `DistilBERT` (English)
- `MacBERT` (Chinese)
- `XLM-R` (Multilingual)

Key techniques:

- Fine-tuning on human vs. LLM-generated text
- Mixed-granularity training for robustness
- Cross-domain and cross-LLM evaluation
- Evaluation across LLMs, domains, and input granularities to assess model robustness
- Custom benchmark dataset: **AGT-2025**

## 🧪 Evaluation

- Models were evaluated using **precision**, **recall**, **F1**, and **accuracy**
- Emphasis on **generalization**: unseen domains, granularities, and LLMs
- Key findings:
  - RoBERTa achieves F1 > 0.99 on GPT-4.5
  - All detectors fail to reliably detect DeepSeek R1 in Chinese (F1 < 0.30)
  - XLM-R shows better robustness in Chinese cross-LLM scenarios

## 📁 Repository Structure

```plaintext
├── src/                    # Code for training, evaluation, preprocessing
│   ├── models/             # Model wrappers and training scripts
│   ├── data/               # Dataset loading and processing
│   ├── analysis/           # Evaluation metrics and plots
│   └── utils/              # Helper functions
├── notebooks/              # Jupyter notebooks for analysis
├── results/                # Logs, predictions, confusion matrices
├── figures/                # Visualizations used in thesis
├── appendix/               # Supplementary tables and stats
├── Detecting-MGT-Thesis.pdf  # Full thesis document
└── README.md               # This file
