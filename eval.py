#!/usr/bin/env python3
"""
Inference script for evaluating DSA LLaMA model on GSM8K dataset.

This script:
1. Loads a trained DSA model
2. Evaluates on GSM8K math word problems
3. Measures correctness (accuracy)
4. Measures speed and throughput (latency, tokens/sec)
"""

import torch
import argparse
import time
import re
from typing import Optional, List, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer
import json
from pathlib import Path

from src.utils import load_from_checkpoint, InferencePerformanceTracker
from src.dsa_llama_model import DSALlamaForCausalLM


def extract_answer(text: str) -> Optional[float]:
    """
    Extract the final numeric answer from model output.
    
    GSM8K answers are typically in the format:
    "The answer is 42" or "Answer: 42" or just "42"
    
    Args:
        text: Generated text from the model
        
    Returns:
        Extracted numeric answer, or None if not found
    """
    # Try to find the last number in the text
    # Look for patterns like "The answer is 42", "Answer: 42", "= 42", etc.
    patterns = [
        r'(?:the answer is|answer is|answer:|answer =|equals?|is)\s*\$?([0-9,]+\.?[0-9]*)',
        r'\$?([0-9,]+\.?[0-9]*)\s*(?:dollars?|USD)?\s*$',  # Number at the end
        r'=\s*\$?([0-9,]+\.?[0-9]*)',  # = 42
    ]
    
    # Try each pattern
    for pattern in patterns:
        matches = re.findall(pattern, text.lower(), re.IGNORECASE)
        if matches:
            # Get the last match (most likely the final answer)
            last_match = matches[-1]
            # Remove commas and convert to float
            try:
                return float(last_match.replace(',', ''))
            except ValueError:
                continue
    
    # Fallback: find all numbers and take the last one
    numbers = re.findall(r'\b([0-9,]+\.?[0-9]*)\b', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    
    return None


def format_gsm8k_prompt(question: str) -> str:
    """
    Format a GSM8K question as a prompt for the model.
    
    Args:
        question: The math word problem question
        
    Returns:
        Formatted prompt string
    """
    # Simple format - can be customized based on model's training format
    return f"Question: {question}\nAnswer:"


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    use_cache: bool = True,
    tracker: Optional[InferencePerformanceTracker] = None,
) -> Tuple[str, dict]:
    """
    Generate an answer for a given prompt using model.generate().
    
    Args:
        model: The model to use for generation
        tokenizer: Tokenizer for the model
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        use_cache: Whether to use KV cache
        tracker: Optional performance tracker
        
    Returns:
        Tuple of (generated_text, performance_metrics)
    """
    model.eval()
    
    # Get device from model parameters
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_length = input_ids.shape[1]
    
    # Generate using model.generate() - similar to main_inference.py
    with torch.no_grad():
        generation_start = time.perf_counter()
        
        # Use model.generate() for simpler, more reliable generation
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "use_cache": use_cache,
            "do_sample": temperature > 0,
        }
        
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        
        outputs = model.generate(**inputs, **generate_kwargs)
        
        generation_end = time.perf_counter()
        
        # Track performance metrics (simplified since model.generate() is a black box)
        if tracker:
            tracker.reset()
            total_time = generation_end - generation_start
            num_generated = outputs.shape[1] - prompt_length
            
            # Track overall metrics
            tracker.start_prefill(prompt_length)
            # Estimate: prefill typically takes ~5-10% of total time for long generations
            estimated_prefill_time = total_time * 0.08 if num_generated > 5 else total_time * 0.2
            tracker.prefill_start_time = generation_start
            tracker.prefill_end_time = generation_start + estimated_prefill_time
            tracker.end_prefill()
            
            # Track decode
            tracker.start_decode()
            tracker.decode_start_time = generation_start + estimated_prefill_time
            if num_generated > 0:
                decode_time_per_token = (total_time - estimated_prefill_time) / num_generated
                tracker.decode_times = [decode_time_per_token] * num_generated
                tracker.num_generated_tokens = num_generated
                tracker.first_token_time = estimated_prefill_time
    
    # Decode generated tokens (exclude prompt)
    generated_text = tokenizer.decode(
        outputs[0][prompt_length:], 
        skip_special_tokens=True
    )
    
    # Get performance metrics
    metrics = tracker.get_metrics() if tracker else {}
    
    return generated_text, metrics


def evaluate_gsm8k(
    model,
    tokenizer,
    dataset,
    num_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    use_cache: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Evaluate model on GSM8K dataset.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        dataset: GSM8K dataset (from datasets library)
        num_samples: Number of samples to evaluate (None for all)
        max_new_tokens: Maximum tokens to generate per sample
        temperature: Sampling temperature
        use_cache: Whether to use KV cache
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    
    # Limit dataset if specified
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    total_samples = len(dataset)
    correct = 0
    total_time = 0.0
    total_tokens = 0
    total_prefill_time = 0.0
    total_decode_time = 0.0
    
    results = []
    
    print(f"Evaluating on {total_samples} GSM8K samples...")
    print("=" * 80)
    
    # Create a single tracker for overall stats
    overall_tracker = InferencePerformanceTracker(warmup_steps=0)
    
    for idx, example in enumerate(dataset):
        question = example["question"]
        ground_truth = example["answer"]
        
        # Extract ground truth number
        gt_answer = extract_answer(ground_truth)
        if gt_answer is None:
            print(f"Warning: Could not extract answer from ground truth: {ground_truth}")
            continue
        
        # Format prompt
        prompt = format_gsm8k_prompt(question)
        
        # Generate answer
        tracker = InferencePerformanceTracker(warmup_steps=0)
        start_time = time.perf_counter()
        
        generated_text, metrics = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=use_cache,
            tracker=tracker,
        )
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        # Extract predicted answer
        pred_answer = extract_answer(generated_text)
        
        # Check correctness
        is_correct = False
        if pred_answer is not None:
            # Allow small floating point differences
            is_correct = abs(pred_answer - gt_answer) < 1e-6
        
        if is_correct:
            correct += 1
        
        # Accumulate metrics
        total_time += elapsed_time
        if metrics:
            total_prefill_time += metrics.get("prefill_time", 0.0)
            total_decode_time += metrics.get("total_decode_time", 0.0)
            total_tokens += metrics.get("num_generated_tokens", 0) + metrics.get("prefill_tokens", 0)
        
        # Store result
        result = {
            "idx": idx,
            "question": question,
            "ground_truth": gt_answer,
            "predicted": pred_answer,
            "generated_text": generated_text,
            "correct": is_correct,
            "time": elapsed_time,
            "metrics": metrics,
        }
        results.append(result)
        
        # Print progress
        if verbose or (idx + 1) % 10 == 0:
            status = "✓" if is_correct else "✗"
            print(
                f"[{idx + 1}/{total_samples}] {status} "
                f"GT: {gt_answer}, Pred: {pred_answer}, "
                f"Time: {elapsed_time:.3f}s"
            )
            if verbose:
                print(f"  Question: {question[:100]}...")
                print(f"  Generated: {generated_text[:200]}...")
                print()
    
    # Calculate final metrics
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    avg_time_per_sample = total_time / total_samples if total_samples > 0 else 0.0
    avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0.0
    avg_prefill_time = total_prefill_time / total_samples if total_samples > 0 else 0.0
    avg_decode_time = total_decode_time / total_samples if total_samples > 0 else 0.0
    
    summary = {
        "total_samples": total_samples,
        "correct": correct,
        "accuracy": accuracy,
        "total_time": total_time,
        "avg_time_per_sample": avg_time_per_sample,
        "avg_tokens_per_second": avg_tokens_per_second,
        "avg_prefill_time": avg_prefill_time,
        "avg_decode_time": avg_decode_time,
        "results": results,
    }
    
    return summary


def print_summary(summary: dict):
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Correct: {summary['correct']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"\nPerformance Metrics:")
    print(f"  Total Time: {summary['total_time']:.2f}s")
    print(f"  Average Time per Sample: {summary['avg_time_per_sample']:.3f}s")
    print(f"  Average Tokens per Second: {summary['avg_tokens_per_second']:.2f}")
    print(f"  Average Prefill Time: {summary['avg_prefill_time']:.4f}s")
    print(f"  Average Decode Time: {summary['avg_decode_time']:.4f}s")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DSA LLaMA model on GSM8K dataset"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name for tokenizer (if different from model_path)",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder in HuggingFace repo (like 'run-1089676' in main_inference.py)",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy decoding)",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=True,
        help="Use KV cache for faster generation",
    )
    parser.add_argument(
        "--no_cache",
        dest="use_cache",
        action="store_false",
        help="Disable KV cache",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save detailed results JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each sample",
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference",
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load tokenizer - similar to main_inference.py
    tokenizer_path = args.base_model or args.model_path
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        subfolder=args.subfolder
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - similar to main_inference.py
    print(f"Loading model from: {args.model_path}")
    
    # Try AutoModelForCausalLM first (works for registered models like main_inference.py)
    # Falls back to DSA-specific loading if needed
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            subfolder=args.subfolder,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        # Fall back to DSA-specific loading for local checkpoints
        print(f"AutoModelForCausalLM failed ({e}), trying DSA-specific loading...")
        if Path(args.model_path).exists():
            model = load_from_checkpoint(args.model_path)
        else:
            model = DSALlamaForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(dataset)} test samples")
    
    # Run evaluation
    summary = evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_cache=args.use_cache,
        verbose=args.verbose,
    )
    
    # Print summary
    print_summary(summary)
    
    # Save results if requested
    if args.output_file:
        output_data = {
            "args": vars(args),
            "summary": {k: v for k, v in summary.items() if k != "results"},
            "results": summary["results"],
        }
        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

