from dataclasses import dataclass

@dataclass
class Config:
    model_name = {
        "roberta_sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest"
    }

    train_path = "/kaggle/input/semevaldataset/Dataset/train_subtask2a.csv"
    output_dir = "./semeval_subtask2a_model"

    window_size = 8
    max_seq_length = 512
    per_device_train_batch_size = 16
    gradient_accumulation_steps = 2

    num_train_epochs = 8
    learning_rate = 2e-5
    weight_decay = 0.08
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.1
    logging_steps = 50
    seed = 3407
    max_grad_norm = 1.0

    fp16 = torch.cuda.is_available()
    bf16 = False
    optim = "adamw_torch"
    seed = 3407