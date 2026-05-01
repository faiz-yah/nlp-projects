# app/inference.py
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

FINETUNED_MODEL_ID = "fabienyah321/agribot-flan-t5"  # ← change this
BASE_MODEL_ID      = "google/flan-t5-base"
MAX_INPUT_LEN      = 256
MAX_TARGET_LEN     = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    model.eval()
    return model, tokenizer

def generate(question: str, model, tokenizer, max_new_tokens=128) -> str:
    prompt = f"answer farming question: {question}"
    inputs = tokenizer(prompt, return_tensors="pt",
                       max_length=MAX_INPUT_LEN, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)