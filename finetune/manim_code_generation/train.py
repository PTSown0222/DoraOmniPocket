import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from manim_code_generation.processing import data_split, formatting_prompts_func
from finetune.process_data import load_hf_dataset
from .cli import parse_cli_args

# init model and tokenizer
def init_tokenizer_and_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"[INFO] Model loaded: {config.base_model}")
    print(f"[INFO] Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    return model, tokenizer

# lora config
def get_lora(model, config):
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print(f"[INFO] PEFT initialized with r={config.lora_r}, alpha={config.lora_alpha}")
    return model


def main():
    config = parse_cli_args()
    print(f"[INFO] Running training with output directory: {config.output_dir}")

    data = load_hf_dataset(config.input_dir, split="train")
    train_ds, val_ds = data_split(data)
    
    # Truyền config vào hàm khởi tạo
    model, tokenizer = init_tokenizer_and_model(config)
    model = get_lora(model, config)
    
    tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
                "{{ 'System: ' + message['content'] + '\n\n' }}"
            "{% elif message['role'] == 'user' %}"
                "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ ' ' + message['content'] + eos_token }}"
            "{% endif %}"
        "{% endfor %}"
    )

    train_data = train_ds.map(formatting_prompts_func, batched=True)
    val_data = val_ds.map(formatting_prompts_func, batched=True)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    model.config.use_cache = False

    training_arguments = SFTConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_batch_size, 
        gradient_accumulation_steps=config.grad_accum_steps,
        num_train_epochs=config.n_epochs,
        max_grad_norm=config.max_grad_norm,

        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        eval_accumulation_steps=1,
        max_steps=-1,

        optim="paged_adamw_32bit",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        dataset_text_field="text",
        max_length=config.max_length,
        lr_scheduler_type="cosine",

        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        fp16=False,
        bf16=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        processing_class=tokenizer,
        args=training_arguments,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()


