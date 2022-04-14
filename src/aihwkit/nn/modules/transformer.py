"""Analog Bert Transformer Module
Adapted from:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
"""

from typing import Optional, Tuple

from math import sqrt

from torch import Tensor, FloatTensor
from torch import concat, arange, matmul, einsum, long
from torch.nn import Dropout, Embedding, LayerNorm, ReLU, SiLU, GELU, Tanh, Sigmoid
from torch.nn.functional import softmax

from aihwkit.nn import AnalogLinear, AnalogSequential

ACT2FN = {
    "relu": ReLU(),
    "silu": SiLU(),
    "swish": SiLU(),
    "gelu": GELU(),
    "tanh": Tanh(),
    "sigmoid": Sigmoid()
}

class AnalogBertSelfAttention(AnalogSequential):
    """Analog Bert Self Attention Module"""

    def __init__(self, config, position_embedding_type=None):
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

        self.query = AnalogLinear(config.hidden_size, self.all_head_size)
        self.key = AnalogLinear(config.hidden_size, self.all_head_size)
        self.value = AnalogLinear(config.hidden_size, self.all_head_size)

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
            key_layer = concat([past_key_value[0], key_layer], dim=2)
            value_layer = concat([past_key_value[1], value_layer], dim=2)
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

    def __init__(self, config):
        super().__init__()
        self.dense = AnalogLinear(config.hidden_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """Forward pass of output from Self Attention"""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(AnalogSequential):
    """The full Self Attention Module with Self Output modules"""

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = AnalogBertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = AnalogBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
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


class BertIntermediate(AnalogSequential):
    def __init__(self, config):
        super().__init__()
        self.dense = AnalogLinear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(AnalogSequential):
    def __init__(self, config):
        super().__init__()
        self.dense = AnalogLinear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
