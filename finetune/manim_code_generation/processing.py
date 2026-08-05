# code_generation/manin_code_dataset.py
import argparse
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

def data_split(ds, split=0.8, seed=42):
    split_ds = ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]
    print(f"Train samples: {len(train_ds)} | Eval samples: {len(val_ds)}")
    print("Sample message sequence:", train_ds[0]["instruction"])
    return train_ds, val_ds
    

def formatting_prompts_func(examples):
    instructions = examples.get("instruction", [])
    inputs = examples.get("input", [""] * len(instructions))
    outputs = examples.get("output", [])

    texts = []

    for instr, inp, out in zip(instructions, inputs, outputs):
        user_content = f"{instr}\nInput: {inp}".strip() if (inp and str(inp).strip()) else instr

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": out}
        ]

        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False, 
            add_generation_prompt=False 
        )
        texts.append(formatted_text)

    return {"text": texts}