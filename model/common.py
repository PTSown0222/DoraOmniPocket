"""
Common utilities for donutchat

bf16 requires SM 80+ (Ampere: A100, A10, etc)
Olders GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
"""

import os

_DTYPE_MAP = {
    "bfloat": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32
}

def _detect_compute_dtype():
    env = os.environ.get("MiniAI")

    if env is not None:
        return _DTYPE_MAP[env], f"set through MINIAI_TYPE={env}"
    
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        
def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("CHAT_BASE_URL"):
        donutchat_dir = os.environ.get("CHAT_BASE_URL")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        donutchat_dir = os.path.join(cache_dir, "donutchat")
    
    os.makedirs(donutchat_dir,exist_ok = True)
    return donutchat_dir
