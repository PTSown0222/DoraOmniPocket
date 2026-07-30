"""Main Training Pipeline"""
import torch
import torch.nn as nn

# local import
from config import Config
from optim import CCCLoss
from dataset import fix_spacing, prepare_data, EmotionDatasetSubtask2a
from model import AttentionPooling, Expert, SparseMoELayer, ValenceAndArousalModel

def train_model():
    config = Config()
    print(">>> [1] Loading & Processing Data...")

    if not os.path.exists(Config.train_path):
        print(f"ERROR: File not found at {Config.train_path}")
        return

    # Load & Prepare Data
    df_processed = prepare_data(Config.train_path)

    # Split Train/Test (Group Shuffle Split)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.seed)
    train_idx, test_idx = next(splitter.split(df_processed, groups=df_processed['user_id']))

    print(f"Total Samples: {len(df_processed)}")
    print(f"Train Samples: {len(train_idx)} | Test Samples: {len(test_idx)}")

    print(">>> [2] Tokenizing & Creating Datasets...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name['roberta_sentiment'])

    # Tạo Dataset từ DataFrame đã chia
    train_ds = EmotionDatasetSubtask2a(df_processed.iloc[train_idx], tokenizer)
    test_ds = EmotionDatasetSubtask2a(df_processed.iloc[test_idx], tokenizer)

    print(">>> [3] Initializing Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Subtask2aModel(Config.model_name['roberta_sentiment']).to(device)

    # Args
    training_args = TrainingArguments(
        output_dir=Config.output_dir,
        learning_rate=Config.learning_rate,
        per_device_train_batch_size=Config.per_device_train_batch_size,
        per_device_eval_batch_size=Config.per_device_train_batch_size * 2,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,

        num_train_epochs=Config.num_train_epochs,
        warmup_ratio=Config.warmup_ratio,

        # Speed Optimization
        group_by_length=True,
        dataloader_num_workers=2,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="avg_r",
        greater_is_better=True,

        fp16=Config.fp16,
        save_total_limit=1,
        remove_unused_columns=False,

        report_to="none",
        logging_steps=Config.logging_steps,

        weight_decay=Config.weight_decay,
        lr_scheduler_type=Config.lr_scheduler_type,
        max_grad_norm=Config.max_grad_norm,
        optim=Config.optim,

        save_safetensors=True,
        seed=Config.seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_subtask2a,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print(">>> [4] Starting Training...")
    trainer.train()
    plot_final_version_results(trainer)

    print(">>> [5] Saving BEST Model for Inference...")
    save_path = "./final_subtask2a_model"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    model.config.save_pretrained(save_path)
    trainer.save_state()
    
    # Zip model to download
    try:
        shutil.make_archive("/kaggle/working/final_subtask2a_model", 'zip', out_path)
        print("Zipped to final_model.zip")
    except: pass

    print(">>> [6] Evaluation on Test Set...")
    eval_metrics = trainer.evaluate(test_ds)
    
    print("=" * 40)
    print(f"RESULTS SUBTASK 2A:")
    print(f"Avg Pearson R: {eval_metrics['eval_avg_r']:.4f}")
    print(f"  - Valence R: {eval_metrics['eval_r_v']:.4f}")
    print(f"  - Arousal R: {eval_metrics['eval_r_a']:.4f}")
    print(f"Avg MAE: {eval_metrics['eval_avg_mae']:.4f}")
    print(f"  - mae Valence: {eval_metrics['eval_mae_v']:.4f}")
    print(f"  - mae Arousal: {eval_metrics['eval_mae_a']:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    train_model()