from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.utils import InferencePerformanceTracker
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

# Use the existing input_ids as the initial sequence
input_ids = inputs['input_ids']
# Get the batch size and the initial length of the prompt
batch_size, initial_len = input_ids.shape
# Define max length
max_length = initial_len + 128
# A token ID to indicate the sequence should stop (e.g., EOS token ID)
eos_token_id = model.config.eos_token_id

tracker = InferencePerformanceTracker(0)

with torch.no_grad():
    tracker.start_decode()
    for step in range(128): # max_new_tokens
        
        # outputs = model(input_ids) # Simple call
        outputs = model(input_ids=input_ids, attention_mask=inputs.get('attention_mask'))
        
        next_token_logits = outputs.logits[:, -1, :]
        tracker.step(len(next_token_logits))
        
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1) # shape (batch_size, 1)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        if 'attention_mask' in inputs:
             new_attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)
             inputs['attention_mask'] = torch.cat([inputs['attention_mask'], new_attention_mask], dim=-1)

        if (next_token == eos_token_id).all():
            break
            
# The final generated output is in 'input_ids'
final_output = input_ids

# print(tokenizer.decode(final_output[0], skip_special_tokens=True))


print(tracker.get_summary())