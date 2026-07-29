#!/usr/bin/env python
# coding=utf-8

"""
UNLP 2025 Shared Task: Manipulation Technique Classification
Multilabel classification using a LLaMA-like generative model + LoRA for parameter-efficient training

1) Reads train.parquet with columns: [id, content, techniques, ...]
2) Creates a multi-label classification dataset
3) (Optional) Weighted classes, but disabled by default
4) Fine-tunes with standard Hugging Face Trainer (BCEWithLogits is auto for multi_label_classification)
5) Evaluates with macro-F1
6) Applies LoRA with high rank (256) to reduce VRAM usage
7) Exports final predictions for test.csv
"""

output_dir="/data/llama3_ft_nopara"


import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model

#########################################
# 1) Config & Label Setup
#########################################

# Example generative model – replace with your LLaMA 3 / Gemma model:
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_NAME = "meta-llama/Llama-Guard-3-8B"
TECHNIQUE_LABELS = [
    "straw_man",
    "appeal_to_fear",
    "fud",
    "bandwagon",
    "whataboutism",
    "loaded_language",
    "glittering_generalities",
    "euphoria",
    "cherry_picking",
    "cliche"
]
label2id = {label: i for i, label in enumerate(TECHNIQUE_LABELS)}
id2label = {i: label for label, i in label2id.items()}

#########################################
# 2) Read & Prepare Training Data
#########################################

def load_training_data(train_path="train.parquet"):
    """Load training set, return as pandas DataFrame."""
    df = pd.read_parquet(train_path)
    return df

def process_row_for_multilabel(techniques_list):
    """
    Convert a row's technique list into a binary vector 
    matching the order of TECHNIQUE_LABELS.
    """
    vec = np.zeros(len(TECHNIQUE_LABELS), dtype=int)
    for t in techniques_list:
        if t in label2id:
            vec[label2id[t]] = 1
    return vec

print("Loading train dataset from train.parquet ...")
train_df = load_training_data()  # => columns: [id, content, techniques, ...]

def fix_tech_list(tech_array):
    if tech_array is None:
        return []
    if isinstance(tech_array,list):
        return [str(item) for item in tech_array]
    else:
        return [str(item) for item in tech_array.tolist()]


train_df["techniques"] = train_df["techniques"].apply(fix_tech_list)
train_df = train_df.dropna(subset=["content"]).reset_index(drop=True)

train_df["label_vector"] = train_df["techniques"].apply(process_row_for_multilabel)

ext_dataset_path = "OpenBabylon/unlp2025_40k_generated_Gemini2_Lenta_FreeStyleRandnArticles_ds"  # Update this!
ext_dataset = pd.DataFrame(load_dataset(ext_dataset_path)["train"])
ext_dataset["techniques"] = ext_dataset["techniques"].apply(fix_tech_list)
ext_dataset["id"] = ext_dataset["id"].astype(str) 
ext_dataset = ext_dataset.dropna(subset=["content"]).reset_index(drop=True)
ext_dataset["label_vector"] = ext_dataset["techniques"].apply(process_row_for_multilabel)


label_matrix = np.stack(train_df["label_vector"].values)
class_counts = label_matrix.sum(axis=0)
print("Class counts:", class_counts)

ext_label_matrix= np.stack(ext_dataset["label_vector"].values)
ext_class_counts = ext_label_matrix.sum(axis=0)
print("Extended Class counts:", ext_class_counts)

#########################################
# 3) Create Dataset
#########################################

train_dataset = Dataset.from_pandas(train_df[["id", "content", "label_vector"]])
ext_train_dataset =  Dataset.from_pandas(ext_dataset[["id", "content", "label_vector"]])
ds = train_dataset.train_test_split(test_size=0.2, seed=42)
augmented_train_dataset = concatenate_datasets([ds["train"], ext_train_dataset])
datasets = DatasetDict({"train": augmented_train_dataset, "validation": ds["test"]})
# datasets = DatasetDict({"train":  ds["train"], "validation": ds["test"]})


#########################################
# 4) Tokenize
#########################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# For LLaMA-based models, set pad_token if needed
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

technique_list_str = "\n".join(f"- {label}" for label in TECHNIQUE_LABELS)

def tokenize_fn(example):
    prompts = [
        f"""Given the following text, identify the manipulation techniques used from the following list:
{technique_list_str}

Text: "{content}"

Techniques:"""
        for content in example["content"]
    ]
    return tokenizer(prompts, truncation=True, max_length=1024)

def format_labels(example):
    example["labels"] = np.array(example["label_vector"], dtype=np.float32)
    return example

datasets = datasets.map(tokenize_fn, batched=True)
datasets = datasets.map(format_labels)

columns_to_remove = ["content", "label_vector"]
datasets = datasets.remove_columns(columns_to_remove)

#########################################
# 5) Define Classification Model w/LoRA
#########################################

lora_r = 128
lora_alpha = 64
lora_dropout = 0.05

# On LLaMA-based models, you might want these target modules:
target_modules = ["q_proj","k_proj","v_proj","o_proj",
                  "gate_proj","down_proj","up_proj"]

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="SEQ_CLS",  # important for classification
    target_modules=target_modules,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(TECHNIQUE_LABELS),
    problem_type="multi_label_classification",  # <--- KEY: BCE for multi-label
    id2label=id2label,
    label2id=label2id,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Ensure correct pad token
model.config.pad_token_id = tokenizer.pad_token_id

#########################################
# 6) Define Multilabel Metrics
#########################################
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # Convert logits to probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    f1_per_class = []
    f1_by_label = {}

    for i, label in enumerate(TECHNIQUE_LABELS):
        f1_c = f1_score(labels[:, i], preds[:, i], zero_division=0)
        f1_per_class.append(f1_c)
        f1_by_label[f"f1_{label}"] = round(f1_c, 4)

    f1_macro = float(np.mean(f1_per_class))

    return {
        "f1_macro": round(f1_macro, 4),
        **f1_by_label
    }
#########################################
# 7) TrainingArguments & Standard Trainer
#########################################
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=25,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=8e-6,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    lr_scheduler_type="linear",
    bf16=True,  # Or fp16 if your GPU doesn't support bf16

)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    

)

#########################################
# 8) Train
#########################################
trainer.train(resume_from_checkpoint="/data/llama3_ft_nopara/checkpoint-5040")
# trainer.train()

#########################################
# 9) Evaluate on Validation
#########################################
metrics = trainer.evaluate()
print("Validation metrics:", metrics)

#########################################
# 10) Inference on Test Set & Submission
#########################################

def predict_test(test_path="test.csv", output_file="/data/llama_ft_unlp/submission.csv"):
    # Load test
    test_df = pd.read_csv(test_path)
    test_ds = Dataset.from_pandas(test_df[["id", "content"]].dropna().reset_index(drop=True))

    # Tokenize
    test_ds = test_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.remove_columns(["content"])

    # Predict
    preds_logits = trainer.predict(test_ds).predictions
    probs = 1.0 / (1.0 + np.exp(-preds_logits))
    pred_labels = (probs >= 0.5).astype(int)

    # Convert to DataFrame
    pred_df = pd.DataFrame(pred_labels, columns=TECHNIQUE_LABELS)
    pred_df.insert(0, "id", test_ds["id"])

    submission_cols = ["id"] + TECHNIQUE_LABELS
    pred_df = pred_df[submission_cols]
    pred_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")

# Example usage:
predict_test("test.csv", f"{output_dir}/submission.csv")
print("Done!")