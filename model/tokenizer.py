"""
BPE Tokenizer in the style of GPT-4.
Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) using RustBPE for training and tiktoken for inference tokenizer
"""
import os
import tiktoken
import torch
from functools import lru_cache # use cache to store memory to avoid being expensive computation

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# reference for GPT-4
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceWrapper:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    @classmethod
    def from_pretrained(cls, hf_path):
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)
    
    @classmethod
    def from_directory(cls, tokenizer_dir):
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)
    
    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = HFTokenizer(BPE(
            byte_fallback = True,
            unk_token = None,
            fuse_unk = False
        ))
        # normalize
        tokenizer.normalizer = None
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern = gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space = False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None

        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )

        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens
    
    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)
