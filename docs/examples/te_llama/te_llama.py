# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import re
import gc
from contextlib import contextmanager

import torch
from torch import nn

import transformer_engine as te
from transformer_engine.pytorch.dot_product_attention.rope import RotaryPositionEmbedding
from transformer_engine.pytorch.fp8 import fp8_model_init
from transformer_engine.pytorch.linear_cross_entropy import linear_cross_entropy

import transformers
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers.modeling_utils import _add_variant, load_state_dict, _load_state_dict_into_model
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils.hub import get_checkpoint_shard_files


@contextmanager
def replace_decoder(te_decoder_cls):
    """
    Replace `LlamaDecoderLayer` with custom `TELlamaDecoderLayer`.
    """
    original_llama_decoder_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = original_llama_decoder_cls


class TELlamaDecoderLayer(te.pytorch.TransformerLayer):
    """
    Wrapper class over TE's `TransformerLayer`. This makes the wrapper very
    similar to HF's `LlamaDecoderLayer` and easier to replace it in the code.

    Args:
        config: LlamaConfig
        args: positional args (for compatibility with `LlamaDecoderLayer`)
        kwargs: keyword args (for compatibility with `LlamaDecoderLayer`)
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=False,
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            num_gqa_groups=config.num_key_value_heads,
        )
        te_rope = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self, hidden_states, *args, attention_mask, **kwargs):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `LlamaDecoderLayer`.
        """
        return (
            super().forward(
                hidden_states, attention_mask=attention_mask, rotary_pos_emb=self.te_rope_emb
            ),
        )


class TELlamaForCausalLM:
    """
    Causal LM created with `LlamaModel`. The underlying `LlamaDecoderLayer`
    class is monkey-patched with `TELlamaDecoderLayer` class before
    initializing the causal LM with `LlamaForCausalLM`.

    Args:
        config: LlamaConfig
    """

    def __new__(cls, config: LlamaConfig):
        with replace_decoder(te_decoder_cls=TELlamaDecoderLayer):
            llama_for_causal_lm = LlamaForCausalLM(config)
        return llama_for_causal_lm

    @classmethod
    def from_pretrained_local(cls, pretrained_model_name_or_path, *args, config, **kwargs):
        """
        Custom method adapted from `from_pretrained` method in HuggingFace
        Transformers repo: https://github.com/huggingface/transformers/blob/f497f564bb76697edab09184a252fc1b1a326d1e/src/transformers/modeling_utils.py#L2579
        """
        # Before loading the model, set the default dtype for torch
        torch.set_default_dtype(kwargs["torch_dtype"])

        # Load the vanilla model weights
        vanilla_model = cls(config)
        subfolder = ""
        variant = None
        if os.path.isfile(
            os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant("model.safetensors.index.json", variant),
            )
        ):
            # Load from a sharded PyTorch checkpoint
            archive_file = os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant("model.safetensors.index.json", variant),
            )
            is_sharded = True
        elif os.path.isfile(
            os.path.join(
                pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
            )
        ):
            # Load from a sharded PyTorch checkpoint
            archive_file = os.path.join(
                pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
            )
            is_sharded = True
        else:
            raise AssertionError("Only sharded PyTorch ckpt format supported at the moment")

        resolved_archive_file, _ = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            archive_file,
        )

        # If the checkpoint is not sharded, it's a trivial sharding case
        if not is_sharded:
            assert not isinstance(resolved_archive_file, list)
            resolved_archive_file = [resolved_archive_file]

        for shard_file in resolved_archive_file:
            state_dict = load_state_dict(shard_file)
            # replace_params copies parameters relevant only to TransformerEngine
            replace_params(state_dict, vanilla_model.state_dict(), config)
            # _load_state_dict_into_model copies parameters other than those in TransformerEngine
            _load_state_dict_into_model(vanilla_model, state_dict, start_prefix="")

            # Force mem release. Taken from huggingface code
            del state_dict
            gc.collect()

        return vanilla_model


def replace_params(hf_state_dict, te_state_dict, config):
    # collect all layer prefixes to update
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        layer_prefix_pat = "model.layers.\d+."
        m = re.match(layer_prefix_pat, param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())

    for layer_prefix in all_layer_prefixes:
        # When loading weights into models with less number of layers, skip the
        # copy if the corresponding layer doesn't exist in HF model
        if layer_prefix + "input_layernorm.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.layernorm_qkv.layer_norm_weight"].data[
                :
            ] = hf_state_dict[layer_prefix + "input_layernorm.weight"].data[:]

        if layer_prefix + "self_attn.q_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.layernorm_qkv.query_weight"].data[:] = (
                hf_state_dict[layer_prefix + "self_attn.q_proj.weight"].data[:]
            )

        if layer_prefix + "self_attn.k_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.layernorm_qkv.key_weight"].data[:] = (
                hf_state_dict[layer_prefix + "self_attn.k_proj.weight"].data[:]
            )

        if layer_prefix + "self_attn.v_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.layernorm_qkv.value_weight"].data[:] = (
                hf_state_dict[layer_prefix + "self_attn.v_proj.weight"].data[:]
            )

        if layer_prefix + "self_attn.o_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.proj.weight"].data[:] = hf_state_dict[
                layer_prefix + "self_attn.o_proj.weight"
            ].data[:]

        if layer_prefix + "post_attention_layernorm.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.layer_norm_weight"].data[:] = hf_state_dict[
                layer_prefix + "post_attention_layernorm.weight"
            ].data[:]

        # It may happen that gate_proj.weight and up_proj.weight will be in the different files, so we need to
        # load them separately.
        if layer_prefix + "mlp.gate_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[
                : config.intermediate_size
            ] = hf_state_dict[layer_prefix + "mlp.gate_proj.weight"].data

        if layer_prefix + "mlp.up_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[
                config.intermediate_size :
            ] = hf_state_dict[layer_prefix + "mlp.up_proj.weight"].data

        if layer_prefix + "mlp.down_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc2_weight"].data[:] = hf_state_dict[
                layer_prefix + "mlp.down_proj.weight"
            ].data[:]
    return all_layer_prefixes


class TELlamaForCausalLMWithFusedCrossEntropy(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, 
                input_ids = None,
                attention_mask = None,
                position_ids = None,
                past_key_values = None,
                inputs_embeds = None,
                labels = None,
                use_cache = None,
                output_attentions = None,
                output_hidden_states = None,
                return_dict = None,
                cache_position = None,
                logits_to_keep = 0,
                **kwargs):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = model_outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]

        loss = None
        logits = None
        if labels is not None:
            with torch.cuda.nvtx.range("fused_loss"):
                lm_head_weight = self.lm_head.weight

                ignore_index = -100
                _labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
                shift_labels = _labels[..., 1:].contiguous()

                batch_size, seq_length, hidden_size = hidden_states.size()

                hidden_states_view = hidden_states.view(-1, hidden_size)
                labels_view = shift_labels.view(-1)
                lm_head_weight_view = lm_head_weight.T.contiguous()

                loss = linear_cross_entropy(
                    hidden_states_view,
                    lm_head_weight_view,
                    labels_view,
                    "mean",
                    None,
                    ignore_index)

            # vanilla implementation
            # with torch.cuda.nvtx.range("vanilla_loss"):
            #     _logits = self.lm_head(hidden_states)
            #     _loss = self.loss_function(logits=_logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            #     loss = _loss
            
        else:
            logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )

class TELlamaForCausalLMWithFusedCrossEntropyFactory(TELlamaForCausalLM):
    def __new__(cls, config: LlamaConfig):
        with replace_decoder(te_decoder_cls=TELlamaDecoderLayer):
            llama_for_causal_lm = TELlamaForCausalLMWithFusedCrossEntropy(config)
        return llama_for_causal_lm