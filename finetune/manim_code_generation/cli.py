# finetune/manim_code_generation/cli.py
import argparse
from .config import ManimFinetuneConfig

def parse_cli_args() -> ManimFinetuneConfig:
    default_cfg = ManimFinetuneConfig()

    parser = argparse.ArgumentParser(description="CLI Finetune Manim Code Generation")

    # Model & Paths
    parser.add_argument("--ver", type=int, default=default_cfg.ver, help="Version number")
    parser.add_argument("--base_model", type=str, default=default_cfg.base_model, help="Base model identifier")
    parser.add_argument("--input_dir", type=str, default=default_cfg.input_dir, help="Dataset path")
    parser.add_argument("--output_dir", type=str, default=default_cfg.output_dir, help="Output directory")
    parser.add_argument("--seed", type=int, default=default_cfg.seed, help="Random seed")

    # Training Hyperparameters
    parser.add_argument("--max_length", type=int, default=default_cfg.max_length)
    parser.add_argument("--learning_rate", type=float, default=default_cfg.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=default_cfg.weight_decay)
    parser.add_argument("--per_device_batch_size", type=int, default=default_cfg.per_device_batch_size)
    parser.add_argument("--grad_accum_steps", type=int, default=default_cfg.grad_accum_steps)
    parser.add_argument("--n_epochs", type=int, default=default_cfg.n_epochs)
    parser.add_argument("--max_grad_norm", type=float, default=default_cfg.max_grad_norm)
    parser.add_argument("--warmup_steps", type=float, default=default_cfg.warmup_steps)

    # LoRA Parameters
    parser.add_argument("--lora_r", type=int, default=default_cfg.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=default_cfg.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=default_cfg.lora_dropout)

    # 1. Parse CLI arguments
    args = parser.parse_args()

    # 2. convert Namespace into Dictionary
    args_dict = vars(args)

    # 3. Unpack dict into ManimFinetuneConfig dataclass
    config = ManimFinetuneConfig(**args_dict)

    return config

if __name__ == "__main__":
    cfg = parse_cli_args()
    print(cfg)