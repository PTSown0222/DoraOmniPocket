from fastapi import FastAPI
from transformers import AutoProcessor, LlavaForConditionalGeneration 
from pydantic import BaseModel
from PIL import Image
import requests
import torch


app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ""
processor = ""

