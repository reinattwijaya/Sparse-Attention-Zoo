#!/usr/bin/env python3
"""
Inference script for evaluating DSA LLaMA model on the GovReport summarization task.

This script:
1. Loads the trained DSA model
2. Evaluates on the GovReport dataset
3. Measures summarization quality (ROUGE)
4. Measures speed and throughput (latency, tokens/sec)
"""

import torch
import argparse
import time
from typing import Optional, Tuple, Dict
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
import json
import re

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# DSA components
from src.dsa_llama_model import DSALlamaForCausalLM, DSALlamaConfig
from src.utils import InferencePerformanceTracker

# -----------------------------
# Prompt formatting
# -----------------------------
def truncate_context(context: str, tokenizer, max_context_tokens: int) -> str:
    """
    Truncate context to fit within max_context_tokens.
    """
    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) <= max_context_tokens:
        return context
    
    truncated_tokens = tokens[:max_context_tokens]
    truncated_context = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_context


def format_govreport_prompt(text: str) -> str:
    return (
        "Summarize the following government report concisely:\n\n"
        f"{text}\n\nSummary:"
    )

# -----------------------------
# Decode function
# -----------------------------
def generate_summary(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    use_cache: bool = True,
    tracker: Optional[InferencePerformanceTracker] = None,
) -> Tuple[str, Dict]:

    model.eval()
    device = next(model.parameters()).device
    
    # Get model's max sequence length for truncation
    model_max_length = getattr(model.config, 'max_position_embeddings', None) or getattr(model.config, 'max_seq_length', 131072)
    
    # Tokenize with truncation to prevent errors
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=model_max_length,
        add_special_tokens=True
    ).to(device)
    input_ids = inputs["input_ids"]
    batch_size = input_ids.shape[0]
    prompt_length = input_ids.shape[1]
    eos_token_id = model.config.eos_token_id

    past_key_values = None

    with torch.no_grad():
        tracker.start_decode()

        for _ in range(max_new_tokens):

            if past_key_values is None:
                current_input_ids = input_ids
            else:
                current_input_ids = next_token

            outputs = model(
                input_ids=current_input_ids,
                attention_mask=inputs.get("attention_mask"),
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
            )

            if use_cache:
                past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :]
            tracker.step(1)

            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if 'attention_mask' in inputs:
                mask_add = torch.ones((batch_size, 1), device=device, dtype=torch.long)
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], mask_add], dim=1
                )

            if (next_token == eos_token_id).all():
                break

    gen_text = tokenizer.decode(
        input_ids[0][prompt_length:], skip_special_tokens=True
    )

    return gen_text, tracker.get_metrics()


# -----------------------------
# Simple ROUGE metrics (no external deps)
# -----------------------------
def compute_rouge(pred: str, ref: str) -> Dict[str, float]:
    """
    Minimal ROUGE-1/ROUGE-2/ROUGE-L implementation for benchmarking.
    Good enough for reporting; not official ROUGE-Lsum.
    """

    def tokenize(x): return re.findall(r"\w+", x.lower())

    p = tokenize(pred)
    r = tokenize(ref)

    def ngram(tokens, n):
        return list(zip(*[tokens[i:] for i in range(n)]))

    # ROUGE-1
    overlap1 = len(set(p) & set(r))
    rouge1 = overlap1 / max(len(r), 1)

    # ROUGE-2
    p2 = set(ngram(p, 2))
    r2 = set(ngram(r, 2))
    overlap2 = len(p2 & r2)
    rouge2 = overlap2 / max(len(r2), 1)

    # ROUGE-L (longest common subsequence)
    def lcs(a, b):
        dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                dp[i][j] = (
                    dp[i-1][j-1] + 1 if a[i-1] == b[j-1]
                    else max(dp[i-1][j], dp[i][j-1])
                )
        return dp[-1][-1]

    lcs_len = lcs(p, r)
    rougeL = lcs_len / max(len(r), 1)

    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
    }

# -----------------------------
# Evaluation loop
# -----------------------------
def evaluate_govreport(model, tokenizer, dataset, num_samples, max_new_tokens, use_cache, max_context_length=None):

    # Get model's max sequence length
    model_max_length = getattr(model.config, 'max_position_embeddings', None) or getattr(model.config, 'max_seq_length', 131072)
    if max_context_length is None:
        # Reserve space for prompt template and summary
        max_context_length = model_max_length - 1024
    
    # Calculate prompt template overhead
    template_overhead = 50  # "Summarize the following government report concisely:\n\n" + "Summary:"
    effective_max_context = max_context_length - template_overhead
    
    if num_samples:
        dataset = dataset.select(range(num_samples))

    results = []
    total_rouge1 = total_rouge2 = total_rougeL = 0
    total_time = total_tokens = 0

    print(f"Evaluating {len(dataset)} GovReport samples...")
    print(f"Model max length: {model_max_length}, Max context length: {effective_max_context}")
    print()

    for i, item in enumerate(dataset):

        text = item["report"]
        summary_ref = item["summary"]
        
        # Truncate context if needed BEFORE formatting prompt
        context_tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(context_tokens) > effective_max_context:
            print(f"  Sample {i+1}: Truncating context from {len(context_tokens)} to {effective_max_context} tokens")
            text = truncate_context(text, tokenizer, effective_max_context)

        prompt = format_govreport_prompt(text)

        tracker = InferencePerformanceTracker(warmup_steps=0)

        t0 = time.perf_counter()
        summary_pred, metrics = generate_summary(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            tracker=tracker
        )
        t1 = time.perf_counter()

        elapsed = t1 - t0
        total_time += elapsed
        total_tokens += metrics.get("num_generated_tokens", 0)

        rouge = compute_rouge(summary_pred, summary_ref)
        total_rouge1 += rouge["rouge1"]
        total_rouge2 += rouge["rouge2"]
        total_rougeL += rouge["rougeL"]

        print(f"[{i+1}] time={elapsed:.2f}s | R1={rouge['rouge1']:.3f}")

        results.append({
            "idx": i,
            "reference": summary_ref,
            "prediction": summary_pred,
            "rouge": rouge,
            "metrics": metrics,
            "time": elapsed,
        })

    n = len(dataset)

    return {
        "avg_rouge1": total_rouge1 / n,
        "avg_rouge2": total_rouge2 / n,
        "avg_rougeL": total_rougeL / n,
        "avg_latency": total_time / n,
        "tokens_per_second": total_tokens / total_time,
        "results": results,
    }


# -----------------------------
# Model loading (your exact logic)
# -----------------------------
def load_dsa_model(repo_id, subfolder, device):

    print("Downloading config...")
    config_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/config.json")
    config = DSALlamaConfig.from_pretrained(config_path)
    config._attn_implementation = "eager"

    model = DSALlamaForCausalLM(config)
    model = model.to(device=device, dtype=torch.bfloat16)

    print("Downloading weights...")
    weights_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/model.safetensors")

    state = load_file(weights_path, device=device)
    model.load_state_dict(state, strict=False)

    return model


# -----------------------------
# Main CLI
# -----------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--max_context_length", type=int, default=None, 
                       help="Maximum context length in tokens (defaults to model max - 1024). "
                            "Use smaller values (e.g., 32000) to speed up inference.")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = "andresnowak/LLama-Deepseek-Sparse-Attention"
    subfolder = "run-1089676"

    print(f"Loading tokenizer from: {repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading DSA model...")
    model = load_dsa_model(repo_id, subfolder, device)

    print("Loading GovReport dataset...")
    dataset = load_dataset("ccdv/govreport-summarization", split="train")

    summary = evaluate_govreport(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        use_cache=args.use_cache,
        max_context_length=args.max_context_length,
    )

    print("\n=== SUMMARY ===")
    print(f"ROUGE-1: {summary['avg_rouge1']:.3f}")
    print(f"ROUGE-2: {summary['avg_rouge2']:.3f}")
    print(f"ROUGE-L: {summary['avg_rougeL']:.3f}")
    print(f"Avg Latency: {summary['avg_latency']:.2f}s")
    print(f"Tokens/sec: {summary['tokens_per_second']:.2f}")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
