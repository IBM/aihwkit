"""Analog Bert Transformer Module"""

from typing import Optional, Tuple

from math import sqrt

from torch import Tensor, FloatTensor
from torch import concat, arange, einsum, long
from torch.nn.functional import softmax

from aihwkit.nn import AnalogLinear
from aihwkit.nn.modules.base import AnalogModuleBase

class BertSelfAttention(AnalogModuleBase):
    """
        Analog Bert Self Attention Module
    """
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config.hidden_size, config.hidden_size, False)
        if (config.hidden_size % config.num_attention_heads != 0
            and not hasattr(config, "embedding_size")):
            raise ValueError(
                f"The hidden size ({config.hidden_size})"
                f" is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = AnalogLinear(config.hidden_size, self.all_head_size)
        self.key = AnalogLinear(config.hidden_size, self.all_head_size)
        self.value = AnalogLinear(config.hidden_size, self.all_head_size)

        # Store the outputs from layers to use in intermediate matrix multiplication
        self.query_out = AnalogLinear(self.all_head_size, self.all_head_size, bias=False)
        self.attention_out = AnalogLinear(self.all_head_size, self.all_head_size, bias=False)

        # TODO: Implement Dropout
        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        # TODO: Implement Embedding
        # if self.position_embedding_type == "relative_key" 
        # or self.position_embedding_type == "relative_key_query":
        #     self.max_position_embeddings = config.max_position_embeddings
        #     self.distance_embedding = nn.Embedding(2 * 
        #                               config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        """Transpose the matrix to prepare for creation of attention scores"""
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
        """Forward pass for the self attention module"""

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
        self.query_out.set_weights(query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(Tensor, Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(Tensor, Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = self.query_out(key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = arange(seq_length, dtype=long, device=hidden_states.device).view(-1, 1)
            position_ids_r = arange(seq_length, dtype=long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype)  # fp16 compatibility

            # TODO: Convert einsums using reshapes and transposing
            if self.position_embedding_type == "relative_key":
                relative_position_scores = einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        self.attention_out.set_weights(attention_probs)
        context_layer = self.attention_out(value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
