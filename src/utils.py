import torch
from transformers import AutoModelForCausalLM
import time

from .dsa_llama_model import DSALlamaForCausalLM, DSALlamaConfig

def load_from_checkpoint(model_path: str):
    model = DSALlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )
    model.config._attn_implementation = "eager"

    return model


def create_dsa_llama_model_from_scratch(
    model_path: str,
    index_top_k: int,
    index_num_heads: int,
    rope_head_dim: int,
    index_head_dim: int):
    # Use official LLaMA 3.2 1B config (architecture only)
    config = DSALlamaConfig.from_pretrained(
        model_path,         
        index_top_k=index_top_k,
        index_num_heads=index_num_heads,
        rope_head_dim=rope_head_dim,
        index_head_dim=index_head_dim,
        )
    
    config._attn_implementation="eager"
    
    print("Model config:", config)

    # Build model from config (NO WEIGHTS LOADED)
    model = DSALlamaForCausalLM(config).to(torch.bfloat16)

    return model


def create_dsa_llama_model_pretrained(
    model_path: str,
    index_top_k: int,
    index_num_heads: int,
    rope_head_dim: int,
    index_head_dim: int,
    subfolder: str = None):
    # Load config with your custom parameters
    config = DSALlamaConfig.from_pretrained(
        model_path,         
        index_top_k=index_top_k,
        index_num_heads=index_num_heads,
        rope_head_dim=rope_head_dim,
        index_head_dim=index_head_dim
    )
    config._attn_implementation="eager"

    # Create model with config
    model = DSALlamaForCausalLM(config).to(torch.bfloat16)
    
    # Load pretrained weights from original Llama
    from transformers import LlamaForCausalLM
    pretrained_model = LlamaForCausalLM.from_pretrained(
        model_path,
        subfolder=subfolder,
        torch_dtype=torch.bfloat16
    )
    
    # Copy matching weights (embeddings, decoder layers, lm_head)
    model.model.embed_tokens.load_state_dict(pretrained_model.model.embed_tokens.state_dict())
    model.model.norm.load_state_dict(pretrained_model.model.norm.state_dict())
    model.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())
    
    # Load weights for each layer (except indexer which is new)
    for layer_idx in range(config.num_hidden_layers):
        # Load everything except self_attn (which has indexer)
        model.model.layers[layer_idx].input_layernorm.load_state_dict(
            pretrained_model.model.layers[layer_idx].input_layernorm.state_dict()
        )
        model.model.layers[layer_idx].post_attention_layernorm.load_state_dict(
            pretrained_model.model.layers[layer_idx].post_attention_layernorm.state_dict()
        )
        model.model.layers[layer_idx].mlp.load_state_dict(
            pretrained_model.model.layers[layer_idx].mlp.state_dict()
        )
        
        # Load attention weights (q, k, v, o projections)
        model.model.layers[layer_idx].self_attn.q_proj.load_state_dict(
            pretrained_model.model.layers[layer_idx].self_attn.q_proj.state_dict()
        )
        model.model.layers[layer_idx].self_attn.k_proj.load_state_dict(
            pretrained_model.model.layers[layer_idx].self_attn.k_proj.state_dict()
        )
        model.model.layers[layer_idx].self_attn.v_proj.load_state_dict(
            pretrained_model.model.layers[layer_idx].self_attn.v_proj.state_dict()
        )
        model.model.layers[layer_idx].self_attn.o_proj.load_state_dict(
            pretrained_model.model.layers[layer_idx].self_attn.o_proj.state_dict()
        )
        # Note: indexer weights remain randomly initialized
    
    del pretrained_model  # Free memory
    
    return model


# From accelerate/examples
class PerformanceTracker:
    """Track training performance metrics."""

    def __init__(self, warmup_steps: int = 10):
        self.warmup_steps = warmup_steps
        self.reset()

    def reset(self):
        """Reset all tracking variables."""
        self.start_time = None
        self.num_tokens = 0
        self.is_in_warmup = True
        self.step_count = 0

    def step(self, batch_tokens: int, model_flops_per_token: float | None = None) -> dict:
        """
        Update performance tracking with a new step.

        Args:
            batch_tokens (int): Number of tokens in current batch

        Returns:
            dict: Performance metrics if past warmup, empty dict otherwise
        """
        self.step_count += 1

        if self.step_count == self.warmup_steps:
            self.start_time = time.perf_counter()
            self.num_tokens = 0
            self.is_in_warmup = False
            return {"warmup_completed": True}

        if not self.is_in_warmup and self.start_time is not None:
            dct = {}
            self.num_tokens += batch_tokens
            total_time = time.perf_counter() - self.start_time
            steps_from_warmup = self.step_count - self.warmup_steps

            if total_time > 0 and steps_from_warmup > 0:
                dct = {
                    "tokens_per_second": self.num_tokens / total_time,
                    "steps_per_second": steps_from_warmup / total_time,
                    "total_tokens": self.num_tokens,
                    "total_time": total_time,
                }

            if model_flops_per_token is not None:
                flops = model_flops_per_token * self.num_tokens
                dct["tflops_per_device"] = flops / (total_time * 1e12)

            return dct

        return {}
    

def get_model_flops_per_token(model: AutoModelForCausalLM, seq_len: int) -> float:
    """
    Get the number of flops per token for the model.

    Args:
        model (AutoModelForCausalLM): Model to get the flops for
        seq_len (int): Sequence length
    """

    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    # MLP: 3 matmuls
    mlp_flops = 18 * cfg.hidden_size * cfg.intermediate_size

    # Indexer: 3 projections + attention computation
    # q_proj: hidden_size -> num_heads * head_dim (6 FLOPs per element)
    # k_proj: hidden_size -> head_dim (6 FLOPs per element)
    # w_proj: hidden_size -> num_heads (6 FLOPs per element)
    # indexer_proj_flops = 6 * cfg.hidden_size * (index_num_heads * index_head_dim + index_head_dim + index_num_heads)
    # # Attention: q @ k^T scales with sequence length (12 * num_heads * head_dim * seq_len)
    # indexer_attn_flops = 12 * index_num_heads * index_head_dim * seq_len
    # indexer_flops = indexer_proj_flops + indexer_attn_flops

    # Attn (w/o dotproduct)
    attn_flops = 12 * head_dim * (cfg.num_attention_heads + cfg.num_key_value_heads)

    # attn (dotproduct) - this scales quadratically with sequence length
    attn_dotproduct_flops = 12 * cfg.num_attention_heads * head_dim * seq_len

    # we also ignore embeddings and layernorms, indexer, etc
    return (mlp_flops + attn_flops + attn_dotproduct_flops) * cfg.num_hidden_layers


def get_model_size_breakdown(model: torch.nn.Module) -> str:
    """
    Generate a detailed model size breakdown string.

    Args:
        model: The model to analyze

    Returns:
        str: Formatted string with model size breakdown
    """
    lines = []

    def format_params(n):
        """Format parameter count in millions and billions."""
        if n >= 1e9:
            return f"{n / 1e9:.3f}B"
        else:
            return f"{n / 1e6:.2f}M"

    # Get parameter counts per module
    module_params = {}
    for name, param in model.named_parameters():
        parts = name.split('.')
        # Group by layer and component
        if len(parts) >= 3 and parts[1] == 'layers':
            layer_idx = parts[2]
            component = '.'.join(parts[3:-1])  # e.g., 'self_attn.q_proj'
            key = f"layers.{layer_idx}.{component}"
        else:
            key = '.'.join(parts[:-1])  # Everything except the final weight/bias

        if key not in module_params:
            module_params[key] = 0
        module_params[key] += param.numel()

    total_params = sum(module_params.values())

    lines.append("="*80)
    lines.append(f"Model Size: {format_params(total_params)} ({total_params:,} params)")
    lines.append("="*80)

    # Group by layer
    from collections import defaultdict
    layers_dict = defaultdict(dict)
    non_layer_params = {}

    for key, count in sorted(module_params.items()):
        if key.startswith('model.layers.'):
            parts = key.split('.')
            layer_idx = int(parts[2])
            component = '.'.join(parts[3:])
            layers_dict[layer_idx][component] = count
        else:
            non_layer_params[key] = count

    # Print non-layer params
    if non_layer_params:
        lines.append("\nNon-layer parameters:")
        for key, count in sorted(non_layer_params.items()):
            lines.append(f"  {key}: {format_params(count)}")

    # Print per-layer breakdown
    if layers_dict:
        lines.append(f"\nPer-layer breakdown ({len(layers_dict)} layers):")
        lines.append("-"*80)

        for layer_idx in sorted(layers_dict.keys()):
            components = layers_dict[layer_idx]
            layer_total = sum(components.values())
            lines.append(f"\n[Layer {layer_idx}] Total: {format_params(layer_total)}")

            for component, count in sorted(components.items()):
                lines.append(f"  {component}: {format_params(count)}")

    lines.append("\n" + "="*80)

    return '\n'.join(lines)


class TrainingMetrics:
    """Handles metric aggregation and logging across distributed training ranks."""

    def __init__(self, accelerator, perf_tracker, model_flops_per_token, total_steps=None):
        self.accelerator = accelerator
        self.perf_tracker = perf_tracker
        self.model_flops_per_token = model_flops_per_token
        self.total_tokens = 0
        self.global_step = 0
        self.total_steps = total_steps
        self.start_time = time.time()

    def step(
        self,
        batch,
        ce_loss,
        outputs,
        epoch,
        scheduler,
        iter_time,
        indexer_grad_norm=None,
        main_grad_norm=None,
        baseline_experiment=False,
        warmup_stage=False,
        log_every=1,
    ):
        """
        Compute, aggregate, and optionally log metrics for a training step.
        All ranks must call this (contains collective operations).
        """
        batch_tokens = batch["input_ids"].numel()

        # Aggregate performance metrics
        perf_metrics_local = self.perf_tracker.step(batch_tokens, self.model_flops_per_token)

        # Aggregate batch tokens across all ranks
        batch_tokens_all = self.accelerator.gather(torch.tensor(batch_tokens, device=self.accelerator.device)).sum()

        # Update global counters
        self.total_tokens += batch_tokens_all.item()
        self.global_step += 1

        # Reduce losses across ranks
        ce_loss_reduced = self.accelerator.reduce(ce_loss.detach(), reduction="mean")

        kl_losses_reduced = []
        if not baseline_experiment and "kl_loss" in outputs:
            for layer_kl_loss in outputs["kl_loss"]:
                kl_reduced = self.accelerator.reduce(layer_kl_loss.detach(), reduction="mean")
                kl_losses_reduced.append(kl_reduced.item())

        if perf_metrics_local:
            tokens_per_sec_tensor = torch.tensor(
                perf_metrics_local.get("tokens_per_second", 0.0),
                device=self.accelerator.device
            )
            tflops_per_gpu_tensor = torch.tensor(
                perf_metrics_local.get("tflops_per_device", 0.0),
                device=self.accelerator.device
            )

            # Sum tokens/sec across all GPUs for total throughput
            tokens_per_sec_total = self.accelerator.reduce(tokens_per_sec_tensor, reduction="sum").item()
            # Average TFLOPs per GPU
            tflops_per_gpu_avg = self.accelerator.reduce(tflops_per_gpu_tensor, reduction="mean").item()

            perf_metrics = {
                "tokens_per_second": tokens_per_sec_total,
                "tflops_per_device": tflops_per_gpu_avg
            }
        else:
            perf_metrics = {}

        # Log metrics if it's time
        if self.global_step % log_every == 0:
            self._log_metrics(
                epoch=epoch,
                ce_loss_reduced=ce_loss_reduced,
                kl_losses_reduced=kl_losses_reduced,
                perf_metrics=perf_metrics,
                scheduler=scheduler,
                iter_time=iter_time,
                indexer_grad_norm=indexer_grad_norm,
                main_grad_norm=main_grad_norm,
                baseline_experiment=baseline_experiment,
                warmup_stage=warmup_stage,
            )

        return {
            "ce_loss": ce_loss_reduced,
            "kl_losses": kl_losses_reduced,
            "perf_metrics": perf_metrics,
            "batch_tokens_all": batch_tokens_all.item(),
            "global_step": self.global_step,
            "total_tokens": self.total_tokens,
        }

    def _log_metrics(
        self,
        epoch,
        ce_loss_reduced,
        kl_losses_reduced,
        perf_metrics,
        scheduler,
        iter_time,
        indexer_grad_norm,
        main_grad_norm,
        baseline_experiment,
        warmup_stage,
    ):
        """Internal method to log metrics to wandb and console."""
        log_dict = {
            "train/ce_loss": ce_loss_reduced.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "train/epoch": epoch,
            "train/global_step": self.global_step,
            "train/total_tokens": self.total_tokens,
            "train/iter_time": iter_time,
        }

        if not baseline_experiment and indexer_grad_norm is not None:
            log_dict["train/indexer_grad_norm"] = indexer_grad_norm.item()
        if (not warmup_stage or baseline_experiment) and main_grad_norm is not None:
            log_dict["train/main_grad_norm"] = main_grad_norm.item()

        if not baseline_experiment and kl_losses_reduced:
            # Add per-layer KL losses
            for i, kl_loss in enumerate(kl_losses_reduced):
                log_dict[f"train/kl_loss_layer_{i}"] = kl_loss
            log_dict["train/mean_kl_loss"] = sum(kl_losses_reduced) / len(kl_losses_reduced)

        # Add performance metrics
        if perf_metrics:
            log_dict["train/tokens_per_sec"] = perf_metrics.get("tokens_per_second", 0)
            log_dict["train/tflops_per_gpu"] = perf_metrics.get("tflops_per_device", 0)

        self.accelerator.log(log_dict, step=self.global_step)

        # Console output (only on main process)
        if self.accelerator.is_main_process:
            perf_str = ""
            if perf_metrics and "tflops_per_device" in perf_metrics:
                perf_str = f" | TFLOPs/GPU: {perf_metrics['tflops_per_device']:.2f}"

            grad_norm_str = ""
            if not baseline_experiment and indexer_grad_norm is not None:
                grad_norm_str = f" | Indexer GradNorm: {indexer_grad_norm.item():.2e}"
            if (not warmup_stage or baseline_experiment) and main_grad_norm is not None:
                grad_norm_str += f" | Main GradNorm: {main_grad_norm.item():.2e}"

            kl_loss_str = ""
            if not baseline_experiment and "train/mean_kl_loss" in log_dict:
                kl_loss_str = f" | Mean KL Loss: {log_dict['train/mean_kl_loss']:.4f}"

            # Calculate ETA
            eta_str = ""
            if self.total_steps is not None and self.global_step > 0:
                elapsed = time.time() - self.start_time
                eta_seconds = (elapsed / self.global_step) * (self.total_steps - self.global_step)
                eta_str = f" | eta (s): {eta_seconds}"

            print(
                f"Epoch {epoch} | Step {self.global_step} | "
                f"CE Loss: {ce_loss_reduced.item():.4f}"
                f"{kl_loss_str} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}{perf_str}{grad_norm_str} | "
                f"Iter Time: {iter_time:.3f}s"
                f"{eta_str}"
            )



class InferencePerformanceTracker:
    """Track inference performance metrics (latency, throughput, etc.)."""

    def __init__(self, warmup_steps: int = 3):
        """
        Initialize inference performance tracker.
        
        Args:
            warmup_steps (int): Number of warmup steps to skip before tracking metrics
        """
        self.warmup_steps = warmup_steps
        self.reset()

    def reset(self):
        """Reset all tracking variables."""
        self.prefill_start_time = None
        self.prefill_end_time = None
        self.decode_start_time = None
        self.decode_times = []
        self.num_generated_tokens = 0
        self.step_count = 0
        self.is_in_warmup = True
        self.first_token_time = None
        self.prefill_tokens = 0

    def start_prefill(self, prompt_tokens: int):
        """
        Mark the start of prefill phase (processing the prompt).
        
        Args:
            prompt_tokens (int): Number of tokens in the prompt
        """
        self.prefill_start_time = time.perf_counter()
        self.prefill_tokens = prompt_tokens

    def end_prefill(self):
        """Mark the end of prefill phase."""
        if self.prefill_start_time is not None:
            self.prefill_end_time = time.perf_counter()

    def start_decode(self):
        """Mark the start of decode phase (generation)."""
        if self.decode_start_time is None:
            self.decode_start_time = time.perf_counter()

    def step(self, num_tokens: int = 1) -> dict:
        """
        Update performance tracking for a decode step.
        
        Args:
            num_tokens (int): Number of tokens generated in this step (usually 1)
            
        Returns:
            dict: Performance metrics if past warmup, empty dict otherwise
        """
        self.step_count += 1
        current_time = time.perf_counter()

        # Track first token time
        if self.first_token_time is None and self.decode_start_time is not None:
            self.first_token_time = current_time - self.decode_start_time

        # Track decode step time
        if self.decode_start_time is not None and self.step_count > self.warmup_steps:
            if len(self.decode_times) == 0:
                # First decode step after warmup
                step_time = current_time - self.decode_start_time
            else:
                # Subsequent steps
                step_time = current_time - (self.decode_start_time + sum(self.decode_times))
            
            if step_time > 0:
                self.decode_times.append(step_time)
                self.num_generated_tokens += num_tokens

        # Return metrics if past warmup
        if self.step_count > self.warmup_steps and len(self.decode_times) > 0:
            return self.get_metrics()
        
        return {}

    def get_metrics(self) -> dict:
        """
        Get current performance metrics.
        
        Returns:
            dict: Dictionary containing all performance metrics
        """
        metrics = {}
        
        # Prefill metrics
        if self.prefill_start_time is not None and self.prefill_end_time is not None:
            prefill_time = self.prefill_end_time - self.prefill_start_time
            metrics["prefill_time"] = prefill_time
            if prefill_time > 0:
                metrics["prefill_tokens_per_second"] = self.prefill_tokens / prefill_time

        # Decode metrics
        if len(self.decode_times) > 0:
            total_decode_time = sum(self.decode_times)
            avg_time_per_token = total_decode_time / len(self.decode_times)
            
            metrics.update({
                "time_to_first_token": self.first_token_time if self.first_token_time is not None else None,
                "time_per_token": avg_time_per_token,
                "tokens_per_second": 1.0 / avg_time_per_token if avg_time_per_token > 0 else 0.0,
                "total_decode_time": total_decode_time,
                "num_generated_tokens": self.num_generated_tokens,
                "num_decode_steps": len(self.decode_times),
            })

        # Total generation metrics
        if self.prefill_start_time is not None and len(self.decode_times) > 0:
            total_time = (self.prefill_end_time if self.prefill_end_time else time.perf_counter()) - self.prefill_start_time
            total_tokens = self.prefill_tokens + self.num_generated_tokens
            metrics.update({
                "total_time": total_time,
                "total_tokens": total_tokens,
                "overall_tokens_per_second": total_tokens / total_time if total_time > 0 else 0.0,
            })

        return metrics

    def get_summary(self) -> str:
        """
        Get a formatted summary string of all metrics.
        
        Returns:
            str: Formatted summary string
        """
        metrics = self.get_metrics()
        if not metrics:
            return "No metrics available yet."
        
        lines = []
        lines.append("=" * 60)
        lines.append("Inference Performance Summary")
        lines.append("=" * 60)
        
        if "prefill_time" in metrics:
            lines.append(f"Prefill Time: {metrics['prefill_time']:.4f}s")
            if "prefill_tokens_per_second" in metrics:
                lines.append(f"Prefill Throughput: {metrics['prefill_tokens_per_second']:.2f} tokens/s")
        
        if "time_to_first_token" in metrics and metrics["time_to_first_token"] is not None:
            lines.append(f"Time to First Token (TTFT): {metrics['time_to_first_token']:.4f}s")
        
        if "time_per_token" in metrics:
            lines.append(f"Time per Token (TPT): {metrics['time_per_token']:.4f}s")
        
        if "tokens_per_second" in metrics:
            lines.append(f"Decode Throughput: {metrics['tokens_per_second']:.2f} tokens/s")
        
        if "num_generated_tokens" in metrics:
            lines.append(f"Generated Tokens: {metrics['num_generated_tokens']}")
        
        if "total_time" in metrics:
            lines.append(f"Total Time: {metrics['total_time']:.4f}s")
            if "overall_tokens_per_second" in metrics:
                lines.append(f"Overall Throughput: {metrics['overall_tokens_per_second']:.2f} tokens/s")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)