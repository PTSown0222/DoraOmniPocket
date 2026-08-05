"""
This module handles loading datasets (Hugging Face Hub or local files)
and processing text/prompts for Fine-tuning Large Language Models (LLMs).
"""

import os
import sys
import pickle
import json
import pandas as pd
from itertools import islice
from typing import Dict, Any, List, Optional, Union

from datasets import load_dataset as hf_load_dataset, Dataset, IterableDataset
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Dataset Loading Functions
# -----------------------------------------------------------------------------

# Load from Hugging Face Hub
def load_hf_dataset(repo_id: str = "thanhkt/manim_code", split: str = "train") -> Dataset:
    """
    Load a dataset from Hugging Face Hub.
    """
    try:
        ds = hf_load_dataset(repo_id, split=split)
        print(f"Successfully loaded HF dataset '{repo_id}' ({split} split).")
        return ds
    except Exception as e:
        print(f"Error loading Hugging Face dataset '{repo_id}': {e}")
        sys.exit(1)

# download from streaming
def load_streaming_dataset(repo_id: str = "thanhkt/manim_code", split: str = "train", num_samples_to_show: int = 3) -> IterableDataset:
    """
    Load a streaming dataset from Hugging Face Hub without downloading to disk.
    """
    try:
        streamed_dataset = hf_load_dataset(repo_id, split=split, streaming=True)
        print(f"Successfully connected to stream: '{repo_id}' ({split} split).")
        
        # Preview a few samples
        dataset_iterator = iter(streamed_dataset)
        print(f"\n--- Previewing top {num_samples_to_show} samples ---")
        for i, example in enumerate(islice(dataset_iterator, num_samples_to_show)):
            text_snippet = str(example.get('input', example.get('text', 'N/A')))[:150] + "..."
            print(f"Sample {i+1}: {text_snippet}")
        print("---------------------------------------------------\n")
        
        return streamed_dataset
    except Exception as e:
        print(f"Error during streaming load: {e}")
        sys.exit(1)

# load from local
def load_local_dataset(file_path: str, format_type: str = "pickle") -> Union[Dataset, pd.DataFrame, Any]:
    """
    Load dataset from a local file (pickle, json, jsonl, csv).
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    try:
        if format_type == "pickle":
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        elif format_type == "json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif format_type == "csv":
            data = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        print(f"Successfully loaded local file: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading local dataset: {e}")
        sys.exit(1)


