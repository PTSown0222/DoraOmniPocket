"""
We only use parquet file for training because it's small and light
"""

import pandas as pd
import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool
from donutchat.common import get_base_dir

base_dir = get_base_dir()
BASE_URL = "..."
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet

def convert_file(url: str, file_name = "text"):
    if file_name == "csv":
        file = pd.to_parqet(url)    
    return file

def list_parquet_file(data_dir = None, warn_on_legacy = False):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir

    if not os.path.exists(data_dir):
        data_dir = os.path.join(base_dir, "base_data")
    
    parquet_files = sorted([
        f for f in os.listdir(data_dir) if f.endwith(".parquet") and not f.endwith(".tmp")
    ])

    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

    
