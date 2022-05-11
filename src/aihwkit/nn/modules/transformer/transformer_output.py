"""ModelOutput classes from huggingface repo"""

from dataclasses import dataclass, fields
from typing import Optional, Tuple, OrderedDict, Any

from torch import FloatTensor, Tensor

class ModelOutput(OrderedDict):
    """Base class for all model outputs as dataclass. Has a `__getitem__` that
    allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes.
    Otherwise behaves like a regular python dictionary.
    <Tip warning={true}>
    You can't unpack a `ModelOutput` directly.
    Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.
    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if len(class_fields) == 0:
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__}"
                             f"should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not isinstance(first_field, Tensor):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                name = getattr(self, field.name)
                if name is not None:
                    self[field.name] = name

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        """Setdefault cannot be used on this class"""
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        """Pop cannot be used on this class"""
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        """Update cannot be used on this class"""
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]

        return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """Convert self to a tuple containing all the attributes/keys that are not `None`."""
        return tuple(self[k] for k in self.keys())

@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence
            (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of
            models, this returns
            the classification token after processing through a linear layer and a tanh activation
            function. The linear
            layer weights are trained from the next sentence prediction (classification) objective
            during pretraining.
        hidden_states (`tuple(FloatTensor)`, *optional*, returned when `output_hidden_states=True`
        is passed or when `config.output_hidden_states=True`):
            Tuple of `FloatTensor` (one for the output of the embeddings,
            if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional
            initial embedding outputs.
        attentions (`tuple(FloatTensor)`, *optional*, returned when `output_attentions=True`
        is passed or when `config.output_attentions=True`):
            Tuple of `FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax,
            used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(FloatTensor)`, *optional*, returned when `output_attentions=True`
        and `config.add_cross_attention=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `FloatTensor` (one for each layer) of
            shape `(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights of the decoder's cross-attention layer,
            after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`tuple(tuple(FloatTensor))`,
        *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(FloatTensor)` of length `config.n_layers`,
            with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states
            (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks)
            that can be used (see `past_key_values`
            input) to speed up sequential decoding.
    """

    last_hidden_state: FloatTensor = None
    pooler_output: FloatTensor = None
    hidden_states: Optional[Tuple[FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[FloatTensor]]] = None
    attentions: Optional[Tuple[FloatTensor]] = None
    cross_attentions: Optional[Tuple[FloatTensor]] = None


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """Base class for model's outputs that may
    also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the
            sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(FloatTensor))`, *optional*, returned when
        `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(FloatTensor)` of length `config.n_layers`, with each
            tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the
            self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks)
            that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(FloatTensor)`, *optional*, returned when `output_hidden_states=True`
        is passed or when `config.output_hidden_states=True`):
            Tuple of `FloatTensor` (one for the output of the embeddings,
            if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the
            optional initial embedding outputs.
        attentions (`tuple(FloatTensor)`, *optional*, returned when `output_attentions=True`
        is passed or when `config.output_attentions=True`):
            Tuple of `FloatTensor` (one for each layer) of
            shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax,
            used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(FloatTensor)`, *optional*, returned when `output_attentions=True`
        and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `FloatTensor` (one for each layer) of
            shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer,
            after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[FloatTensor]]] = None
    hidden_states: Optional[Tuple[FloatTensor]] = None
    attentions: Optional[Tuple[FloatTensor]] = None
    cross_attentions: Optional[Tuple[FloatTensor]] = None
