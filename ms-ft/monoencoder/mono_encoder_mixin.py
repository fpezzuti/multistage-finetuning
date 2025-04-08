import torch
import math
from abc import ABC
from functools import partial
from typing import Callable, Tuple
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertSelfAttention
    
class MonoEncoderMixin(torch.nn.Module, ABC):
    self_attention_pattern = "self"

    def __init__(
        self,
        original_forward: Callable[
            ...,
            Tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions,
        ],
        use_flash: bool,
        depth: int | None,
    ) -> None:
        self.original_forward = original_forward
        self.use_flash = use_flash
        self.depth = depth

    def forward(self, *args, **kwargs):
        attention_forward = (
            self.flash_attention_forward if self.use_flash else self.attention_forward
        )
        for name, module in self.named_modules():
            if name.endswith(self.self_attention_pattern):
                module.forward = partial(
                    attention_forward,
                    module,
                )
        return self.original_forward(self, *args, **kwargs)

    def flash_attention_forward(
        _self,
        self: BertSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: Tuple[Tuple[torch.FloatTensor]] | None = None,
        output_attentions: bool | None = False,
    ) -> Tuple[torch.Tensor]:
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        context = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask.to(query.dtype) if attention_mask is not None else None,
            self.dropout.p if self.training else 0,
        )

        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(new_context_shape)
        return (context,)

    def attention_forward(
        _self,
        self: BertSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: Tuple[Tuple[torch.FloatTensor]] | None = None,
        output_attentions: bool | None = False,
    ) -> Tuple[torch.Tensor]:
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask # apply attention mask

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1) # normalize attention scores to probs.

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs

class BertMonoEncoderMixin(MonoEncoderMixin):
    encoder_name = "bert"

class RoBERTaMonoEncoderMixin(MonoEncoderMixin):
    encoder_name = "roberta"
    
class ElectraMonoEncoderMixin(MonoEncoderMixin):
    encoder_name = "electra"