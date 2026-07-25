import torch
import torch.nn as nn
from transformer import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig
)

def generate(
    trained_vlm,
    tokenizer,
    input_text,
    image,
    max_new_tokens,
):
    text = [{"role": "user", "content": tokenizer.image_token*256 + input_text}]
    pass

def inference():
    pass
