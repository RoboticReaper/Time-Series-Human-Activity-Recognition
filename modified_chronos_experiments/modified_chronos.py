# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>, Caner Turkmen <atturkm@amazon.com>, Lorenzo Stella <stellalo@amazon.com>
# Original source:
# https://github.com/autogluon/autogluon/blob/f57beb26cb769c6e0d484a6af2b89eab8aee73a8/timeseries/src/autogluon/timeseries/models/chronos/pipeline/chronos_bolt.py

import copy
import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.t5.modeling_t5 import (
    ACT2FN,
    T5Config,
    T5LayerNorm,
    T5PreTrainedModel,
    T5Stack,
)
from transformers.utils import ModelOutput

logger = logging.getLogger(__file__)


@dataclass
class ChronosBoltConfig:
    context_length: int
    prediction_length: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: List[float]
    use_reg_token: bool = False


@dataclass
class ChronosBoltOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None


class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to have shape (batch_size, num_channels, sequence_length)
        batch_size, num_channels, length = x.shape
        if length % self.patch_size != 0:
            padding_size = (
                batch_size,
                num_channels,
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(
                size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device
            )
            x = torch.concat((padding, x), dim=-1)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, x.shape[2], -1)
        return x


class InstanceNorm(nn.Module):
    """
    See, also, RevIN. Apply standardization along the last dimension.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            # Step 1: Calculate mean, ignoring NaNs from padding
            loc = torch.nanmean(x, dim=-1, keepdim=True)
            # Step 2: Clamp inf/nan values in loc to 0, just in case
            loc = torch.nan_to_num(loc, nan=0.0)

            # Step 3: Calculate variance, ignoring NaNs
            variance = torch.nanmean((x - loc).square(), dim=-1, keepdim=True)
            # Step 4: CRITICAL - Clamp inf/nan in variance to 0 to prevent explosion
            variance = torch.nan_to_num(variance, nan=0.0)

            # Step 5: Calculate scale with eps inside sqrt for stability
            scale = (variance + self.eps).sqrt()

        else:
            loc, scale = loc_scale

        return (x - loc) / scale, (loc, scale)

    def inverse(
        self, x: torch.Tensor, loc_scale: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        loc, scale = loc_scale
        return x * scale + loc


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class ChronosBoltModelForClassification(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [  # type: ignore
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]  # type: ignore
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]  # type: ignore

    def __init__(self, config: T5Config, num_classes: int, num_channels: int = 1):
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        super().__init__(config)
        self.model_dim = config.d_model
        self.num_classes = num_classes
        self.classification_head = nn.Linear(config.hidden_size, self.num_classes)
        self.num_channels = num_channels
        self.chronos_config = ChronosBoltConfig(**config.chronos_config)

        # Only decoder_start_id (and optionally REG token)
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1

        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        input_embedding_dim = self.chronos_config.input_patch_size * self.num_channels * 2
        self.input_patch_embedding = ResidualBlock(
            in_dim=input_embedding_dim,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )

        self.instance_norm = InstanceNorm()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self._init_decoder(config)

        self.post_init()

        self.model_parallel = False
        self.device_map = None

    def _init_weights(self, module):
        super()._init_weights(module)
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if module is self.input_patch_embedding:
            fan_in = self.chronos_config.input_patch_size * self.num_channels * 2
            std = factor * (fan_in ** -0.5)

            module.hidden_layer.weight.data.normal_(mean=0.0, std=std)
            if module.hidden_layer.bias is not None:
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(mean=0.0, std=std)
            if module.residual_layer.bias is not None:
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(mean=0.0, std=factor * (self.config.d_ff ** -0.5))
            if module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()

        elif module is self.classification_head:
            # Initialize classification head layer
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor * (self.model_dim ** -0.5))
            if module.bias is not None:
                module.bias.data.zero_()

    def encode(self, context: torch.Tensor, mask: Optional[torch.Tensor] = None):

        if context.ndim != 3:
            raise ValueError(f"Context must be a 3D tensor, but got {context.ndim} dimensions")

        if mask is None:
            mask = ~torch.isnan(context)
        mask = mask.to(context.dtype)

        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length:]
            mask = mask[..., -self.chronos_config.context_length:]

        context, loc_scale = self.instance_norm(context)

        context = context.to(self.dtype)

        patched_context = self.patch(context)

        patched_mask = self.patch(mask)

        patched_context = torch.where(patched_mask > 0.0, torch.nan_to_num(patched_context), 0.0)

        patched_mask = torch.nan_to_num(patched_mask, nan=0.0)

        patched_input = torch.cat([patched_context, patched_mask], dim=-1)

        input_embeds = self.input_patch_embedding(patched_input)

        encoder_outputs = self.encoder(
            attention_mask=(patched_mask.sum(dim=-1) > 0).to(torch.int64),
            inputs_embeds=input_embeds,
        )

        return encoder_outputs[0], loc_scale, (patched_mask.sum(dim=-1) > 0).to(torch.int64)

    def forward(self, context: torch.Tensor, mask: Optional[torch.Tensor] = None):
        hidden_states, _, attention_mask = self.encode(context=context, mask=mask)

        batch_size = hidden_states.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1), self.config.decoder_start_token_id, device=hidden_states.device
        )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
        )

        # Get the single output token embedding
        final_hidden_state = decoder_outputs.last_hidden_state.squeeze(1)
        logits = self.classification_head(final_hidden_state)

        return logits

    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

    def decode(
        self,
        input_embeds,
        attention_mask,
        hidden_states,
        output_attentions=False,
    ):
        """
        Parameters
        ----------
        input_embeds: torch.Tensor
            Patched and embedded inputs. Shape (batch_size, patched_context_length, d_model)
        attention_mask: torch.Tensor
            Attention mask for the patched context. Shape (batch_size, patched_context_length), type: torch.int64
        hidden_states: torch.Tensor
            Hidden states returned by the encoder. Shape (batch_size, patched_context_length, d_model)

        Returns
        -------
        last_hidden_state
            Last hidden state returned by the decoder, of shape (batch_size, 1, d_model)
        """
        batch_size = input_embeds.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            device=input_embeds.device,
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return decoder_outputs.last_hidden_state

