"""Analog Bert Transformer Module
Adapted from:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
"""

from math import sqrt

from dataclasses import dataclass

from typing import List, Optional, Tuple, Union

from torch import Tensor, FloatTensor
from torch import concat, arange, matmul, einsum, zeros, ones
from torch import long, utils

from torch.nn import ModuleList, Embedding, LayerNorm, Dropout, Tanh

from torch.nn.functional import softmax

from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    logger,
    load_tf_weights_in_bert,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    BERT_INPUTS_DOCSTRING,
    BERT_START_DOCSTRING,
    _TOKENIZER_FOR_DOC,
    _CONFIG_FOR_DOC,
    _CHECKPOINT_FOR_DOC
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
)
from transformers.utils.generic import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.nn.modules.base import RPUConfigAlias
from aihwkit.simulator.configs.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice

class AnalogBertSelfAttention(AnalogSequential):
    """Analog Bert Self Attention Module"""

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias],
                 realistic_read_write: bool = False,
                 position_embedding_type=None):
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

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias],
                 realistic_read_write: bool):
        super().__init__()
        self.dense = AnalogLinear(config.hidden_size, config.hidden_size,
                                  rpu_config=rpu_config,
                                  realistic_read_write=realistic_read_write)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """Forward pass of output from Self Attention"""
        # pylint: disable=arguments-differ, arguments-renamed
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class AnalogBertAttention(AnalogSequential):
    """The full Self Attention Module with Self Output modules"""

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias],
                 realistic_read_write: bool,
                 position_embedding_type = None):
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
                 realistic_read_write: bool = False):
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

        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """Forward pass of Bert Output"""
        # pylint: disable=arguments-differ, arguments-renamed

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class AnalogBertLayer(AnalogSequential):
    """Analog Bert Layer"""

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias],
                 realistic_read_write: bool = False):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = AnalogBertAttention(
                                config,
                                rpu_config,
                                realistic_read_write=realistic_read_write)

        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model "
                                 "if cross attention is added")
            self.crossattention = AnalogBertAttention(
                                        config,
                                        rpu_config,
                                        realistic_read_write=realistic_read_write,
                                        position_embedding_type="absolute")
        self.intermediate = AnalogBertIntermediate(config,
                                                   rpu_config,
                                                   realistic_read_write=realistic_read_write)
        self.output = AnalogBertOutput(config,
                                       rpu_config,
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

class AnalogBertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

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
            module.weight.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

            # Analog: sync digital parameters with analog tile
            module.reset_parameters()

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


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[FloatTensor] = None
    prediction_logits: FloatTensor = None
    seq_relationship_logits: FloatTensor = None
    hidden_states: Optional[Tuple[FloatTensor]] = None
    attentions: Optional[Tuple[FloatTensor]] = None


BERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class AnalogBertModel(AnalogBertPreTrainedModel, AnalogSequential):
    """The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self,
                 config,
                 rpu_config: Optional[RPUConfigAlias] = None,
                 realistic_read_write: bool = False,
                 add_pooling_layer = True):
        AnalogBertPreTrainedModel.__init__(self, config)
        AnalogSequential.__init__(self)

        self.config = config

        if rpu_config is None:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice())

        self.embeddings = BertEmbeddings(config)
        self.encoder = AnalogBertEncoder(config, rpu_config, realistic_read_write)

        self.pooler = AnalogBertPooler(config, rpu_config, realistic_read_write) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """Get input embeddings"""
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """Set input embeddings"""
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
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
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = ones(((batch_size, seq_length + past_key_values_length)))

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = zeros(input_shape, dtype=long)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
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
