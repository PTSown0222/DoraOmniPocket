# Mistral 7B Python Manim Code Generation

A fine-tuned language model based on the **Mistral-7B-v0.3** architecture, specialized in generating executable **Manim** (Mathematical Animation Engine) Python code from natural language prompts (Text-to-Manim Code).

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-blue)](https://huggingface.co/thanhkt/mistal-7b-codegen)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1g0k5r6J3X8Z7z2Q9F4y5K6L7N8P9Q1R2?usp=sharing)

---

## 📌 Links & Artifacts

* **Hugging Face Hub Weights**: [Mistral Manim Python Coder v0.1](https://huggingface.co/TheSon2202/mistral-manim-python-coder-v01)
* **Training Notebook**: [Google Colab Notebook](https://colab.research.google.com/drive/1msxaUBOXxzJBfKS7OXW6oMfz2PDcuxjJ#scrollTo=auwdeqjhyj1b)
* **Script code here**: [`inference.py`](./inference.py)
---

## 📁 Repository Structure

```text
finetune/manim_code_generation/
├── config.py     # Training configuration schema (Dataclass)
├── cli.py        # Command-line interface argument parser (argparse)
├── train.py      # Entrypoint for LoRA / QLoRA training pipeline
├── infer.py      # Entrypoint for running Manim code inference
└── README.md
```

This repository manages packages and environments using uv. Execute the training script from the root directory of the project
```bash
uv run python -m finetune.manim_code_generation.train \
    --ver 2 \
    --base_model "mistralai/Mistral-7B-v0.3" \
    --per_device_batch_size 4 \
    --grad_accum_steps 2 \
    --learning_rate 1e-4 \
    --n_epochs 3 \
    --lora_r 32
```

Inference

```bash
uv run python -m finetune.manim_code_generation.inference \
    --model_path "./outputs/mistral-7b-v0.3_v2_manim" \
    --prompt "Draw a red square with side length 4 and animate it shifting right by 3 units." \
    --max_new_tokens 768 \
    --temperature 0.2
```



