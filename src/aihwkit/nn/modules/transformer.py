# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Analog Bert Transformer Module
Adapted from:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
"""

import gc
from math import sqrt

from dataclasses import dataclass
import re
import shutil
import tempfile

from typing import List, Optional, Tuple, Union
import warnings

from torch import (
    Tensor,
    FloatTensor,

    cat,
    arange,
    matmul,
    einsum,
    zeros,
    ones,
    full,
    empty,

    long,

    utils,
    device,
)

from torch.nn import (
    Parameter,
    ModuleList,
    Embedding,
    LayerNorm,
    Dropout,
    Tanh,
    CrossEntropyLoss,
    MSELoss,
    BCEWithLogitsLoss,
)

from torch.nn.functional import softmax

from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,

    logger,

    load_tf_weights_in_bert,

    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,

    BERT_INPUTS_DOCSTRING,
    BERT_START_DOCSTRING,

    _TOKENIZER_FOR_DOC,

    _CONFIG_FOR_DOC,

    _CHECKPOINT_FOR_DOC,
    _CHECKPOINT_FOR_QA,
    _CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
    _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,

    _QA_TARGET_START_INDEX,
    _QA_TARGET_END_INDEX,

    _QA_EXPECTED_LOSS,
    _QA_EXPECTED_OUTPUT,
    _TOKEN_CLASS_EXPECTED_OUTPUT,
    _TOKEN_CLASS_EXPECTED_LOSS,
    _SEQ_CLASS_EXPECTED_OUTPUT,
    _SEQ_CLASS_EXPECTED_LOSS,
)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.pytorch_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer,
    apply_chunking_to_forward,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    QuestionAnsweringModelOutput,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    TokenClassifierOutput,
    NextSentencePredictorOutput,
    SequenceClassifierOutput,
)
from transformers.utils.generic import ModelOutput
from transformers.modeling_utils import (
    PreTrainedModel,
    is_accelerate_available,
    load_state_dict,
    _load_state_dict_into_model,
    _load_state_dict_into_meta_model,
)

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.nn.modules.base import RPUConfigAlias
from aihwkit.simulator.configs.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice


if is_accelerate_available():
    from accelerate.utils import (
        load_offloaded_weights,
        save_offload_index,
        set_module_tensor_to_device,
    )


def _add_weights_to_state_dict(module):
    """Register weights for model loading"""
    if isinstance(module, AnalogLinear):
        weight, _ = module.get_weights()
        module.weight = Parameter(weight)

def _sync_analog_digital_weights(module):
    """Set the Analog tile weights to the loaded digital weights
    and unregister digital parameters
    """
    if isinstance(module, AnalogLinear):
        weight = module.weight.data
        bias = module.bias.data

        if module.analog_bias:
            module.unregister_parameter('bias')
        else:
            bias = None

        module.set_weights(weight, bias)

        module.unregister_parameter('weight')


class AnalogBertSelfAttention(AnalogSequential):
    """Analog Bert Self Attention Module"""

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias],
                 realistic_read_write: bool,
                 position_embedding_type: str = None):
        super().__init__()
        if (config.hidden_size % config.num_attention_heads != 0
                and not hasattr(config, "embedding_size")):
            raise ValueError(
                f"The hidden size ({config.hidden_size})"
                f" is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = AnalogLinear(config.hidden_size, self.all_head_size,
                                  rpu_config=rpu_config,
                                  realistic_read_write=realistic_read_write)
        self.key = AnalogLinear(config.hidden_size, self.all_head_size,
                                rpu_config=rpu_config,
                                realistic_read_write=realistic_read_write)
        self.value = AnalogLinear(config.hidden_size, self.all_head_size,
                                  rpu_config=rpu_config,
                                  realistic_read_write=realistic_read_write)

        self.dropout = Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        if (self.position_embedding_type == "relative_key"
                or self.position_embedding_type == "relative_key_query"):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = Embedding(
                2 * config.max_position_embeddings - 1,
                self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        """Prepare matrix for calculating attention scores"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[FloatTensor] = None,
        head_mask: Optional[FloatTensor] = None,
        encoder_hidden_states: Optional[FloatTensor] = None,
        encoder_attention_mask: Optional[FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor]:
        """Forward pass of Self Attention Module"""
        # pylint: disable=arguments-differ, arguments-renamed

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = cat([past_key_value[0], key_layer], dim=2)
            value_layer = cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(Tensor, Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(Tensor, Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to
            # current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = matmul(query_layer, key_layer.transpose(-1, -2))

        if (self.position_embedding_type == "relative_key"
                or self.position_embedding_type == "relative_key_query"):
            seq_length = hidden_states.size()[1]

            position_ids_l = arange(seq_length,
                                    dtype=long,
                                    device=hidden_states.device).view(-1, 1)

            position_ids_r = arange(seq_length,
                                    dtype=long,
                                    device=hidden_states.device).view(1, -1)

            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1)

            # fp16 compatibility
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = einsum("bhld,lrd->bhlr",
                                                  query_layer,
                                                  positional_embedding)
                attention_scores = attention_scores + relative_position_scores

            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = einsum("bhld,lrd->bhlr",
                                                        query_layer,
                                                        positional_embedding)
                relative_position_scores_key = einsum("bhrd,lrd->bhlr",
                                                      key_layer,
                                                      positional_embedding)

                attention_scores = (attention_scores +
                                    relative_position_scores_query +
                                    relative_position_scores_key)

        attention_scores = attention_scores / sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask
            # (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class AnalogBertSelfOutput(AnalogSequential):
    """Output from Self Attention. Applies a Linear, Dropout and LayerNorm to the layers"""

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias],
                 realistic_read_write: bool):
        super().__init__()
        self.dense = AnalogLinear(config.hidden_size, config.hidden_size,
                                  rpu_config=rpu_config,
                                  realistic_read_write=realistic_read_write)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """Forward pass of output from Self Attention"""
        # pylint: disable=arguments-differ, arguments-renamed
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AnalogBertAttention(AnalogSequential):
    """The full Self Attention Module with Self Output modules"""

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias],
                 realistic_read_write: bool,
                 position_embedding_type: str = None):
        super().__init__()
        self.self = AnalogBertSelfAttention(
            config,
            rpu_config,
            realistic_read_write=realistic_read_write,
            position_embedding_type=position_embedding_type)
        self.output = AnalogBertSelfOutput(
            config,
            rpu_config,
            realistic_read_write=realistic_read_write)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """Prune attention heads"""
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[FloatTensor] = None,
        head_mask: Optional[FloatTensor] = None,
        encoder_hidden_states: Optional[FloatTensor] = None,
        encoder_attention_mask: Optional[FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor]:
        """Forward pass of Bert Attention Modules"""
        # pylint: disable=arguments-differ, arguments-renamed

        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class AnalogBertIntermediate(AnalogSequential):
    """Analog Bert Intermediate block"""

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias],
                 realistic_read_write: bool):
        super().__init__()
        self.dense = AnalogLinear(
            config.hidden_size,
            config.intermediate_size,
            rpu_config=rpu_config,
            realistic_read_write=realistic_read_write)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass of Intermediate"""
        # pylint: disable=arguments-differ, arguments-renamed

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class AnalogBertOutput(AnalogSequential):
    """Analog Bert Output block"""

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias],
                 realistic_read_write: bool):
        super().__init__()
        self.dense = AnalogLinear(
            config.intermediate_size,
            config.hidden_size,
            rpu_config=rpu_config,
            realistic_read_write=realistic_read_write)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """Forward pass of Bert Output"""
        # pylint: disable=arguments-differ, arguments-renamed

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AnalogBertLayer(AnalogSequential):
    """Analog Bert Layer"""

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias],
                 realistic_read_write: bool):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = AnalogBertAttention(
            config,
            rpu_config=rpu_config,
            realistic_read_write=realistic_read_write)

        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model "
                                 "if cross attention is added")
            self.crossattention = AnalogBertAttention(
                config,
                rpu_config=rpu_config,
                realistic_read_write=realistic_read_write,
                position_embedding_type="absolute")
        self.intermediate = AnalogBertIntermediate(config,
                                                   rpu_config=rpu_config,
                                                   realistic_read_write=realistic_read_write)
        self.output = AnalogBertOutput(config,
                                       rpu_config=rpu_config,
                                       realistic_read_write=realistic_read_write)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[FloatTensor] = None,
        head_mask: Optional[FloatTensor] = None,
        encoder_hidden_states: Optional[FloatTensor] = None,
        encoder_attention_mask: Optional[FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor]:
        """Forward pass of Analog Bert Layer"""
        # pylint: disable=arguments-differ, arguments-renamed

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # add self attentions if we output attention weights
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, "
                    f"{self} has to be instantiated with cross-attention "
                    f"layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )

            attention_output = cross_attention_outputs[0]
            # add cross attentions if we output attention weights
            outputs = outputs + cross_attention_outputs[1:-1]

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim, attention_output
        )

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """Chunk feed forward, shrink to intermediate size and back"""
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class AnalogBertEncoder(AnalogSequential):
    """Analog Bert Encoder"""

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias],
                 realistic_read_write: bool):
        super().__init__()
        self.config = config
        self.layer = ModuleList([
            AnalogBertLayer(config,
                            rpu_config,
                            realistic_read_write=realistic_read_write)

            for _ in range(config.num_hidden_layers)
        ])

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[FloatTensor] = None,
        head_mask: Optional[FloatTensor] = None,
        encoder_hidden_states: Optional[FloatTensor] = None,
        encoder_attention_mask: Optional[FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        """Forward pass of Encoder/(Decoder) Block"""
        # pylint: disable=arguments-differ, arguments-renamed

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Checkpointing
                if use_cache:
                    # Log Warning
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing."
                        "Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # pylint: disable=cell-var-from-loop
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                # Checkpointing
                layer_outputs = utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class AnalogBertPooler(AnalogSequential):
    """Bert Pooler"""

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias],
                 realistic_read_write: bool):
        super().__init__()
        self.dense = AnalogLinear(
            config.hidden_size,
            config.hidden_size,
            rpu_config=rpu_config,
            realistic_read_write=realistic_read_write)

        self.activation = Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass of pooler"""
        # pylint: disable=arguments-differ, arguments-renamed

        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class AnalogBertPredictionHeadTransform(AnalogSequential):
    """Analog Bert Prediction Head Transform"""

    def __init__(self, config, rpu_config: Optional[RPUConfigAlias], realistic_read_write: bool):
        super().__init__()
        self.dense = AnalogLinear(
            config.hidden_size,
            config.hidden_size,
            rpu_config=rpu_config,
            realistic_read_write=realistic_read_write)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass for AnalogBertPredictionHeadTransform
        Linear -> Hidden Activation -> LayerNorm
        """
        # pylint: disable=arguments-differ, arguments-renamed

        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class AnalogBertLMPredictionHead(AnalogSequential):
    """Analog Bert LM Prediction Head"""

    def __init__(self, config, rpu_config: Optional[RPUConfigAlias], realistic_read_write: bool):
        super().__init__()
        self.transform = AnalogBertPredictionHeadTransform(
            config, rpu_config, realistic_read_write)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = AnalogLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False)

        self.bias = Parameter(zeros(config.vocab_size))

        weight, _ = self.decoder.get_weights()

        # Need a link between the two variables
        # so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.set_weights(weight, self.bias)

    def forward(self, hidden_states):
        """Forward pass of AnalogBertLMPredictionHead
        Performs transform before applying decoder
        """
        # pylint: disable=arguments-differ, arguments-renamed

        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class AnalogBertOnlyMLMHead(AnalogSequential):
    """Analog Bert Only MLM Head"""

    def __init__(self, config, rpu_config: Optional[RPUConfigAlias], realistic_read_write: bool):
        super().__init__()
        self.predictions = AnalogBertLMPredictionHead(config, rpu_config, realistic_read_write)

    def forward(self, sequence_output: Tensor) -> Tensor:
        """Forward pass of AnalogBertOnlyMLMHead"""
        # pylint: disable=arguments-differ, arguments-renamed

        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class AnalogBertOnlyNSPHead(AnalogSequential):
    """Analog Bert Only NSP Head"""

    def __init__(self, config, rpu_config: Optional[RPUConfigAlias], realistic_read_write: bool):
        super().__init__()
        self.seq_relationship = AnalogLinear(
            config.hidden_size,
            2,
            rpu_config=rpu_config,
            realistic_read_write=realistic_read_write)

    def forward(self, pooled_output):
        """Forward pass of AnalogBertOnlyNSPHead"""
        # pylint: disable=arguments-differ, arguments-renamed

        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class AnalogBertPreTrainingHeads(AnalogSequential):
    """Analog Bert Pretraining Heads"""

    def __init__(self, config, rpu_config: Optional[RPUConfigAlias], realistic_read_write: bool):
        super().__init__()
        self.predictions = AnalogBertLMPredictionHead(config, rpu_config, realistic_read_write)
        self.seq_relationship = AnalogLinear(
            config.hidden_size,
            2,
            rpu_config=rpu_config,
            realistic_read_write=realistic_read_write)

    def forward(self, sequence_output, pooled_output):
        """Forward pass of AnalogBertPreTrainingHeads"""
        # pylint: disable=arguments-differ, arguments-renamed

        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class AnalogBertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained
    models.
    """

    # pylint: disable=abstract-method

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, AnalogLinear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            weight, bias = module.get_weights()
            weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if bias is not None:
                bias.data.zero_()

            module.set_weights(weight, bias)

        elif isinstance(module, Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, AnalogBertEncoder):
            module.gradient_checkpointing = value

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
        sharded_metadata=None,
        _fast_init=True,
        low_cpu_mem_usage=False,
        device_map=None,
        offload_folder=None,
        offload_state_dict=False,
        dtype=None,
    ):
        # Analog: add `weight` entry into state_dict() for model loading
        model.apply(_add_weights_to_state_dict)

        if device_map is not None and "disk" in device_map.values() and offload_folder is None:
            raise ValueError(
                "The current `device_map` had weights offloaded to the disk. "
                "Please provide an `offload_folder` for"
                " them."
            )
        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix

        def _fix_key(key):
            if "beta" in key:
                return key.replace("beta", "bias")
            if "gamma" in key:
                return key.replace("gamma", "weight")
            return key

        original_loaded_keys = loaded_keys
        loaded_keys = [_fix_key(key) for key in loaded_keys]

        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(prefix)]
            expected_keys = [".".join(s.split(".")[1:]) if s.startswith(
                prefix) else s for s in expected_keys]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Some models may have keys that are not in the state by design,
        # removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            # pylint: disable=not-an-iterable
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        # retrieve weights on meta device and put them back on CPU.
        # This is not ideal in terms of memory, but if we don't do that not,
        # we can't initialize them in the next step
        if low_cpu_mem_usage:
            for key in missing_keys:
                if key.startswith(prefix):
                    key = ".".join(key.split(".")[1:])
                param = model_state_dict[key]
                if param.device == device("meta"):
                    set_module_tensor_to_device(model, key, "cpu", empty(*param.size()))

        # retrieve unintialized modules and initialize
        # before maybe overriding that with the pretrained weights.
        if _fast_init:
            uninitialized_modules = model.retrieve_modules_from_names(
                missing_keys, add_prefix=add_prefix_to_model, remove_prefix=remove_prefix_from_model
            )
            for module in uninitialized_modules:
                # pylint: disable=protected-access
                model._init_weights(module)

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if (len(cls.base_model_prefix) > 0 and
            not hasattr(model, cls.base_model_prefix) and has_prefix_module):
            start_prefix = cls.base_model_prefix + "."
        if (len(cls.base_model_prefix) > 0 and
            hasattr(model, cls.base_model_prefix) and not has_prefix_module):
            model_to_load = getattr(model, cls.base_model_prefix)
            if any(key in expected_keys_not_prefixed for key in loaded_keys):
                raise ValueError(
                    "The state dictionary of the model you are trying to load is corrupted. "
                    "Are you sure it was "
                    "properly saved?"
                )
            if device_map is not None:
                device_map = {k.replace(f"{cls.base_model_prefix}.", "")
                                        : v for k, v in device_map.items()}

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        # The model key starts with `prefix`
                        # but `checkpoint_key` doesn't so we add it.
                        model_key = f"{prefix}.{checkpoint_key}"
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix`
                        # but `checkpoint_key` does so we remove it.
                        model_key = ".".join(checkpoint_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape,
                             model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict, start_prefix)
        else:
            # Sharded checkpoint or whole but low_cpu_mem_usage==True

            # This should always be a list but, just to be sure.
            if not isinstance(resolved_archive_file, list):
                resolved_archive_file = [resolved_archive_file]

            error_msgs = []
            mismatched_keys = []
            offload_index = {} if device_map is not None and "disk" in device_map.values() else None
            if offload_state_dict:
                state_dict_folder = tempfile.mkdtemp()
                state_dict_index = {}
            else:
                state_dict_folder = None
                state_dict_index = None

            for shard_file in resolved_archive_file:
                state_dict = load_state_dict(shard_file)

                # Mistmatched keys contains tuples key/shape1/shape2 of
                # weights in the checkpoint that have a shape not
                # matching the weights in the model.
                mismatched_keys += _find_mismatched_keys(
                    state_dict,
                    model_state_dict,
                    original_loaded_keys,
                    add_prefix_to_model,
                    remove_prefix_from_model,
                    ignore_mismatched_sizes,
                )

                if low_cpu_mem_usage:
                    new_error_msgs, offload_index, state_dict_index = (
                        _load_state_dict_into_meta_model(
                            model_to_load,
                            state_dict,
                            loaded_keys,
                            start_prefix,
                            expected_keys,
                            device_map=device_map,
                            offload_folder=offload_folder,
                            offload_index=offload_index,
                            state_dict_folder=state_dict_folder,
                            state_dict_index=state_dict_index,
                            dtype=dtype,
                        )
                    )
                    error_msgs += new_error_msgs
                else:
                    error_msgs += _load_state_dict_into_model(
                        model_to_load,
                        state_dict,
                        start_prefix
                    )

                # force memory release
                del state_dict
                gc.collect()

            if offload_index is not None and len(offload_index) > 0:
                save_offload_index(offload_index, offload_folder)

            if offload_state_dict:
                # Load back temporarily offloaded state dict
                load_offloaded_weights(model, state_dict_index, state_dict_folder)
                shutil.rmtree(state_dict_folder)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` "
                    "in the model `from_pretrained` method."
                )
            raise RuntimeError(
                f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}"
            )

        # Analog: Remove missing keys with `analog_ctx` and `analog_tile_state`
        # Since these are generally not going to exist in loaded state_dict's,
        # there is no reason to notify the user that they are not being loaded
        missing_keys = [
            key
            for key in missing_keys
            if 'analog_ctx' not in key and 'analog_tile_state' not in key
        ]

        # pylint: disable=logging-fstring-interpolation
        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were "
                "not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- "
                "This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model "
                "trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification "
                "model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be "
                "exactly identical"
                " (initializing a BertForSequenceClassification model from a "
                "BertForSequenceClassification model)."
            )
        else:
            logger.info(
                "All model checkpoint weights were used when "
                f"initializing {model.__class__.__name__}.\n"
            )
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized "
                "from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\n"
                "You should probably"
                " TRAIN this model on a down-stream task to be "
                "able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were "
                "initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is "
                "similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} "
                "for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the "
                    "model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model "
                "checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized "
                "because the shapes did not"
                f" match:\n{mismatched_warning}\n"
                "You should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        # Analog: set the analog tile weights to the digital weights
        model.apply(_sync_analog_digital_weights)

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """Output type of [`BertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided,
        `FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling
            loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`FloatTensor` of shape
        `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head
            (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification)
            head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(FloatTensor)`, *optional*,
        returned when `output_hidden_states=True` is passed or
        when `config.output_hidden_states=True`):
            Tuple of `FloatTensor` (one for the output of the embeddings +
            one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(FloatTensor)`, *optional*,
        returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax,
            used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[FloatTensor] = None
    prediction_logits: FloatTensor = None
    seq_relationship_logits: FloatTensor = None
    hidden_states: Optional[Tuple[FloatTensor]] = None
    attentions: Optional[Tuple[FloatTensor]] = None


@add_start_docstrings(
    "The bare Analog Bert Model transformer outputting"
    "raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class AnalogBertModel(AnalogBertPreTrainedModel, AnalogSequential):
    """The model can behave as an encoder (with only self-attention)
    as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers,
    following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762)
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with
    the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to
    initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states`
    is then expected as an input to the forward pass.
    """

    # pylint: disable=abstract-method

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias] = None,
                 realistic_read_write: bool = False,
                 add_pooling_layer=True):
        AnalogBertPreTrainedModel.__init__(self, config)
        AnalogSequential.__init__(self)

        self.config = config

        if rpu_config is None:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice())

        self.embeddings = BertEmbeddings(config)
        self.encoder = AnalogBertEncoder(config, rpu_config, realistic_read_write)

        self.pooler = (AnalogBertPooler(config, rpu_config, realistic_read_write)
                       if add_pooling_layer else None)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """Get input embeddings"""
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """Set input embeddings"""
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model. heads_to_prune:
        dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format(
        "batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """encoder_hidden_states
        (`FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask
        (`FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input.
            This mask is used in
            the cross-attention if the model is configured as a decoder.
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(FloatTensor))` of length
        `config.n_layers` with each tuple having 4 tensors of shape
        `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks.
            Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last
            `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape
            `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are
            returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        # pylint: disable=arguments-differ, arguments-renamed

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        input_device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2]
            if past_key_values is not None
            else 0
        )

        if attention_mask is None:
            attention_mask = ones(((batch_size, seq_length + past_key_values_length)))

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size,
                                                                                  seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = zeros(input_shape, dtype=long)

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: Tensor = self.get_extended_attention_mask(attention_mask,
                                                                           input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ones(encoder_hidden_shape, device=input_device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has
        #   shape bsz x n_heads x N x N
        # input head_mask has
        #   shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to
        #   shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """Analog Bert Model with a `language modeling` head on top for CLM fine-tuning.""",
    BERT_START_DOCSTRING
)
class AnalogBertLMHeadModel(AnalogBertPreTrainedModel, AnalogSequential):
    """Analog Bert LM Head Model"""

    # pylint: disable=abstract-method

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(
            self,
            config,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False):
        AnalogBertPreTrainedModel.__init__(self, config)
        AnalogSequential.__init__(self)

        if not config.is_decoder:
            logger.warning(
                "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

        if rpu_config is None:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice())

        self.bert = AnalogBertModel(config, rpu_config, realistic_read_write, False)
        self.cls = AnalogBertOnlyMLMHead(config, rpu_config, realistic_read_write)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """Get output embeddings"""
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings"""
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[List[Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], CausalLMOutputWithCrossAttentions]:
        """encoder_hidden_states  (`FloatTensor` of shape
        `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`FloatTensor` of shape
        `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input.
            This mask is used in
            the cross-attention if the model is configured as a decoder.
            Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring)
            Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with
            labels n `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(FloatTensor))` of length `config.n_layers`
        with each tuple having 4 tensors of shape
        `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks.
            Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only
            the last `decoder_input_ids` (those that
            don't have their past key value states given to this model)
            of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are
            returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        # pylint: disable=arguments-differ, arguments-renamed

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1,
                               self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            **model_kwargs):
        """Prepare inputs for generation"""
        # pylint: disable=unused-argument

        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model,
        # the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx)
                               for past_state in layer_past),)
        return reordered_past


@add_start_docstrings(
    """Analog Bert Model with a `language modeling` head on top.""",
    BERT_START_DOCSTRING
)
class AnalogBertForMaskedLM(AnalogBertPreTrainedModel, AnalogSequential):
    """Analog Bert for Masked LM"""

    # pylint: disable=abstract-method

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(
            self,
            config,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False):
        AnalogBertPreTrainedModel.__init__(self, config)
        AnalogSequential.__init__(self)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        if rpu_config is None:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice())

        self.bert = AnalogBertModel(config, rpu_config, realistic_read_write, False)
        self.cls = AnalogBertOnlyMLMHead(config, rpu_config, realistic_read_write)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """Get the output embeddings"""
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings"""
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.88,
    )
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], MaskedLMOutput]:
        """Labels (`LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should be in
        `[-100, 0, ...,
        config.vocab_size]` (see `input_ids` docstring)
        Tokens with indices set to `-100` are ignored (masked), the
        loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # pylint: disable=arguments-differ, arguments-renamed

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        """Prepare inputs for generation"""
        # pylint: disable=unused-argument

        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=long, device=input_ids.device
        )
        input_ids = cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


@add_start_docstrings(
    """Analog Bert Model with a `next sentence prediction (classification)` head on top.""",
    BERT_START_DOCSTRING,
)
class AnalogBertForNextSentencePrediction(AnalogBertPreTrainedModel, AnalogSequential):
    """Analog Bert for Next Sentence Prediction"""

    # pylint: disable=abstract-method

    def __init__(
            self,
            config,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False):
        AnalogBertPreTrainedModel.__init__(self, config)
        AnalogSequential.__init__(self)

        if rpu_config is None:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice())

        self.bert = AnalogBertModel(config, rpu_config, realistic_read_write)
        self.cls = AnalogBertOnlyNSPHead(config, rpu_config, realistic_read_write)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=NextSentencePredictorOutput,
        config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[Tensor], NextSentencePredictorOutput]:
        """Labels (`LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the next sequence prediction (classification) loss.
        Input should be a sequence pair
        (see `input_ids` docstring). Indices should be in `[0, 1]`:
        - 0 indicates sequence B is a continuation of sequence A,
        - 1 indicates sequence B is a random sequence.

        Returns:
        Example:
        ```python
        >>> from transformers import BertTokenizer, BertForNextSentencePrediction
        >>> import torch
        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant,
        is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")
        >>> outputs = model(**encoding, labels=LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """
        # pylint: disable=arguments-differ, arguments-renamed

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be"
                " removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Analog Bert Model transformer with a sequence classification/regression head on top
    (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class AnalogBertForSequenceClassification(AnalogBertPreTrainedModel, AnalogSequential):
    """Analog Bert For Sequence Classification"""

    # pylint: disable=abstract-method

    def __init__(
            self,
            config,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False):
        AnalogBertPreTrainedModel.__init__(self, config)
        AnalogSequential.__init__(self)
        self.num_labels = config.num_labels
        self.config = config

        if rpu_config is None:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice())

        self.bert = AnalogBertModel(config, rpu_config, realistic_read_write)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.classifier = AnalogLinear(
            config.hidden_size,
            config.num_labels,
            rpu_config=rpu_config,
            realistic_read_write=realistic_read_write)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], SequenceClassifierOutput]:
        """Labels (`LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss.
        Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression
        loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # pylint: disable=arguments-differ, arguments-renamed

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == long or labels.dtype == int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Analog Bert Model with a multiple choice classification head on top
    (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BERT_START_DOCSTRING,
)
class AnalogBertForMultipleChoice(AnalogBertPreTrainedModel, AnalogSequential):
    """Analog Bert For Multiple Choice"""

    # pylint: disable=abstract-method

    def __init__(
            self,
            config,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False):
        AnalogBertPreTrainedModel.__init__(self, config)
        AnalogSequential.__init__(self)

        if rpu_config is None:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice())

        self.bert = AnalogBertModel(config, rpu_config, realistic_read_write)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.classifier = AnalogLinear(
            config.hidden_size,
            1,
            rpu_config=rpu_config,
            realistic_read_write=realistic_read_write)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], MultipleChoiceModelOutput]:
        """Labels (`LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the multiple choice classification loss.
        Indices should be in `[0, ...,
        num_choices-1]` where `num_choices` is the
        size of the second dimension of the input tensors. (See
        `input_ids` above)
        """
        # pylint: disable=arguments-differ, arguments-renamed

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)
                                             ) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)
                                             ) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)
                                         ) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Analog Bert Model with a token classification head on top
    (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class AnalogBertForTokenClassification(AnalogBertPreTrainedModel, AnalogSequential):
    """Analog Bert For Token Classification"""

    # pylint: disable=abstract-method

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(
            self,
            config,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False):
        AnalogBertPreTrainedModel.__init__(self, config)
        AnalogSequential.__init__(self)

        if rpu_config is None:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice())

        self.num_labels = config.num_labels

        self.bert = AnalogBertModel(config, rpu_config, realistic_read_write, False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.classifier = AnalogLinear(
            config.hidden_size,
            config.num_labels,
            rpu_config=rpu_config,
            realistic_read_write=realistic_read_write)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], TokenClassifierOutput]:
        """Labels (`LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the token classification loss. Indices should be
        in `[0, ..., config.num_labels - 1]`.
        """
        # pylint: disable=arguments-differ, arguments-renamed

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Analog Bert Model with a span classification head on top for extractive question-answering tasks
    like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)
class AnalogBertForQuestionAnswering(AnalogBertPreTrainedModel, AnalogSequential):
    """Analog Bert model with Q&A head"""

    # pylint: disable=abstract-method

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(
            self,
            config,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False):

        AnalogBertPreTrainedModel.__init__(self, config)
        AnalogSequential.__init__(self)

        self.num_labels = config.num_labels

        if rpu_config is None:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice())

        self.bert = AnalogBertModel(config, rpu_config, realistic_read_write, False)

        self.qa_outputs = AnalogLinear(
            config.hidden_size,
            config.num_labels,
            rpu_config=rpu_config,
            realistic_read_write=realistic_read_write)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_QA,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=_QA_TARGET_START_INDEX,
        qa_target_end_index=_QA_TARGET_END_INDEX,
        expected_output=_QA_EXPECTED_OUTPUT,
        expected_loss=_QA_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], QuestionAnsweringModelOutput]:
        """start_positions (`LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the
            labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled
            span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # pylint: disable=arguments-differ, arguments-renamed

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
