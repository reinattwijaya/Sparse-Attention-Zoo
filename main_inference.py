from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Base repo
repo = "andresnowak/LLama-Deepseek-Sparse-Attention"

# Choose the run you want
run_subfolder = "run-1089676"  # or "run-1089496"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=run_subfolder)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    repo,
    subfolder=run_subfolder,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
)
model.to(device)
model.eval()

# Inference example
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))