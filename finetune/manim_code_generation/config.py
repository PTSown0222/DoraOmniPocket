# finetune/manim_code_generation/config.py
import os
from dataclasses import dataclass, field

@dataclass
class ManimFinetuneConfig:
    ver: int = 1
    base_model: str = "mistralai/Mistral-7B-v0.3"
    #instruction_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    output_dir: str = ""
    input_dir: str = "Edoh/manim_python"

    #trainer config
    max_length: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 0.03
    per_device_batch: int = 2
    grad_accum_steps: int = 4
    n_epochs: int = 2
    max_grad_norm: float = 0.3
    warmup_steps: float = 0.1

    # lora config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    seed = 2202
    
    def __post_init__(self):
        self.output_dir = f"./mistral_ver{self.ver}_code_generation"
        print(f"create folder at: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)