import torch
import Dataset

class LLMDataset:
    def __init__(self, vocab: str):
        self.text2int = vocab
        self.int2text = {i : }
