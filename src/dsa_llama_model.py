# model.py
from transformers import GenerationMixin
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils.generic import TransformersKwargs, can_return_tuple, check_model_inputs
from transformers.processing_utils import Unpack
from transformers import LlamaConfig, LlamaPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from dataclasses import dataclass
from transformers.models.llama.modeling_llama import LlamaModel, LlamaAttention, apply_rotary_pos_emb, eager_attention_forward, LlamaDecoderLayer, LlamaMLP, LlamaRMSNorm
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from collections.abc import Callable

from .losses import compute_indexer_kl_loss
from .indexer import Indexer


@dataclass
class DSABaseModelOutputWithPast(BaseModelOutputWithPast):
    """
    Extended output with indexer scores.
    """
    indexer_scores: Optional[Tuple[torch.FloatTensor]] = None
    kl_loss: Optional[Tuple[torch.FloatTensor]] = None
    total_kl_loss: Optional[torch.FloatTensor] = None


@dataclass
class DSACausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Extended causal LM output with indexer scores.
    """
    indexer_scores: Optional[Tuple[torch.FloatTensor]] = None
    kl_loss: Optional[Tuple[torch.FloatTensor]] = None
    total_kl_loss: Optional[torch.FloatTensor] = None


class DSALlamaConfig(LlamaConfig):
    model_type = "llama"  # Keep same model type

    def __init__(
        self,
        index_top_k=2048,
        index_num_heads=1,
        index_head_dim=64,
        rope_head_dim=32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.index_top_k = index_top_k
        self.index_num_heads = index_num_heads
        self.rope_head_dim = rope_head_dim
        self.index_head_dim = index_head_dim
    def clone(self):
        return copy.deepcopy(self)


class LlamaDSA(LlamaAttention):
    """
    DeepSeek Sparse Attention (DSA)

    Combines lightning indexer and top-k selection with standard attention mechanism.
    """
    def __init__(self, config: DSALlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.index_top_k = config.index_top_k
        self.rope_head_dim = config.rope_head_dim 
        self.indexer = Indexer(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        warmup_stage: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, seq_len, _ = hidden_states.shape

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (batch_size, num_heads, seq_len, head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        num_query_heads = query_states.shape[1]
        num_kv_heads = key_states.shape[1]
        repetition_factor = num_query_heads // num_kv_heads
        repetitions = (1, repetition_factor, 1, 1)

        cos, sin = position_embeddings

        # Indexer

        # NOTE: Partial Rope in dense attention is from Deeepseek V2/V3 architecure, not DSA
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        # NOTE: (in the paper they say they detach the indexer input from the main computational graph) 
        index_scores = self.indexer(
            hidden_states=hidden_states.detach(),
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # Apply sparse masking only during sparse training (not warmup)

        if not warmup_stage:
            with torch.no_grad():
                if attention_mask is not None:
                    causal_mask = attention_mask[:, 0, :, :]
                    index_scores_masked = index_scores + causal_mask
                else:
                    index_scores_masked = index_scores
                _, top_k_indices = torch.topk(index_scores_masked, k=min(self.index_top_k, index_scores_masked.shape[-1]), dim=-1, sorted=False)
                sparse_mask = torch.full_like(index_scores_masked, -float("inf"))
                sparse_mask = sparse_mask.scatter_(-1, top_k_indices, 0.0) # 0 for the top-k (active) entries; -inf for the rest (deactivated)

                if attention_mask is not None:
                    attention_mask = attention_mask + sparse_mask.unsqueeze(1) # canceling the positions in the attention mask with the -inf from the sparse mask
                else:
                    attention_mask = sparse_mask.unsqueeze(1)

        # ---

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # NOTE: it seems sdpa nor flash_attention return teh attention weights
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # is_q_single = query_states.shape[-2] == 1
        # B, H, L, E = key_states.shape

        # if is_q_single:
        #     active_mask = torch.eq(attention_mask[..., -1,:], 0.0)
        #     active_mask = active_mask.repeat(1, key_states.shape[1], 1)
        #     key_states = key_states[active_mask].reshape(B,H,-1,E)
        #     value_states = value_states[active_mask].reshape(B,H,-1,E)


        attn_weights = None

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            # attention_mask if not is_q_single else None,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # expanded_key_states = key_states.repeat(repetitions).contiguous()
        # expanded_value_states = value_states.repeat(repetitions).contiguous()

        # attn_output = F.scaled_dot_product_attention(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attn_mask=attention_mask,
        #     dropout_p=0.0 if not self.training else self.attention_dropout,
        #     scale=self.scaling,
        #     # is_causal=False, # Set this to True if your model is auto-regressive (decoder-only)
        #     enable_gqa=True,
        # )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, index_scores
    

class DSALlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DSALlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.index_top_k = config.index_top_k

        self.hidden_size = config.hidden_size

                
        self.self_attn = LlamaDSA(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        warmup_stage: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, attn_weights, index_scores = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            warmup_stage=warmup_stage,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights, index_scores
        

class DSALlamaModel(LlamaModel):
    def __init__(self, config: DSALlamaConfig):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [DSALlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        compute_kl_loss: Optional[bool] = False,
        warmup_stage: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DSABaseModelOutputWithPast:
        if use_cache and past_key_values is None:
            # NOTE: if use cache is None, somewhere the cache is created but with only num_hidden_layers size, so we get error in indexer when accessing the cache (because we are accessing an index that doesn't exist)
            # Modify cache to be double size for the indexer
            cache_config = self.config.clone()
            cache_config.num_hidden_layers = self.config.num_hidden_layers * 2  # reserve extra for indexer
            past_key_values = DynamicCache(config=cache_config)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        kl_losses = None
        indexer_scores = None

        total_kl_loss = 0 # The loss we will backpropagate on
        if compute_kl_loss:
            kl_losses = []
            indexer_scores = []

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, attn_weights, index_scores = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values, 
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                warmup_stage=warmup_stage,
                **kwargs,
            )

            # Compute KL loss between indexer and attention
            # - Warmup stage: use full sequence (top_k=None)
            # - Sparse training: use top_k selected tokens
            if compute_kl_loss and attn_weights is not None:
                kl_loss_top_k = None if warmup_stage else self.config.index_top_k
                kl_loss = compute_indexer_kl_loss(attn_weights.detach(), index_scores, top_k=kl_loss_top_k)
                total_kl_loss += kl_loss

                kl_losses.append(kl_loss.detach())
                indexer_scores.append(index_scores.detach())


        hidden_states = self.norm(hidden_states)

        return DSABaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            kl_loss=kl_losses,
            indexer_scores=indexer_scores,
            total_kl_loss=total_kl_loss if compute_kl_loss else None
        )
    
class DSALlamaPreTrainedModel(LlamaPreTrainedModel):
    config: DSALlamaConfig

    _no_split_modules = ["DSALlamaDecoderLayer"]

    _can_record_outputs = {
        "hidden_states": DSALlamaDecoderLayer,
        "attentions": LlamaDSA,
        "indexer_scores": Indexer,
    }


class DSALlamaForCausalLM(DSALlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = DSALlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_main_model(self):
        """Freeze all parameters except indexer (for warmup stage)"""
        for name, param in self.named_parameters():
            if 'indexer' not in name:
                param.requires_grad = False

    def unfreeze_main_model(self):
        """Unfreeze all parameters (for sparse training stage)"""
        for param in self.parameters():
            param.requires_grad = True

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DSACausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        outputs: DSABaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return DSACausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            indexer_scores=outputs.indexer_scores,
            kl_loss=outputs.kl_loss,
            total_kl_loss=outputs.total_kl_loss
        )