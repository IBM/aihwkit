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

"""Utilities for resistive processing units configurations."""

from dataclasses import Field, fields, is_dataclass
from enum import Enum
from textwrap import indent
from typing import Any, List

from aihwkit.simulator.rpu_base import devices, tiles


def parameters_to_bindings(params: Any) -> Any:
    """Convert a dataclass parameter into a bindings class."""
    result = params.bindings_class()
    for field, value in params.__dict__.items():
        # Convert enums to the bindings enums.
        if field in ('unit_cell_devices', 'device', 'mapping'):
            # Exclude special fields that are not present in the bindings.
            continue

        if isinstance(value, Enum):
            if hasattr(tiles, value.__class__.__name__):
                enum_class = getattr(tiles, value.__class__.__name__)
            else:
                enum_class = getattr(devices, value.__class__.__name__)
            enum_value = getattr(enum_class, value.value)
            setattr(result, field, enum_value)
        elif is_dataclass(value):
            setattr(result, field, parameters_to_bindings(value))
        else:
            setattr(result, field, value)

    return result


def tile_parameters_to_bindings(params: Any) -> Any:
    """Convert a tile dataclass parameter into a bindings class."""
    field_map = {'forward': 'forward_io',
                 'backward': 'backward_io'}
    excluded_fields = ('device', 'noise_model', 'drift_compensation',
                       'clip', 'modifier', 'mapping')

    result = params.bindings_class()
    for field, value in params.__dict__.items():
        # Get the mapped field name, if needed.
        field = field_map.get(field, field)

        # Convert enums to the bindings enums.
        if field in excluded_fields:
            # Exclude special fields that are not present in the bindings.
            continue

        if isinstance(value, Enum):
            enum_class = getattr(devices, value.__class__.__name__)
            enum_value = getattr(enum_class, value.value)
            setattr(result, field, enum_value)
        elif is_dataclass(value):
            setattr(result, field, parameters_to_bindings(value))
        else:
            setattr(result, field, value)

    return result


class _PrintableMixin:
    """Helper class for pretty-printing of config dataclasses."""
    # pylint: disable=too-few-public-methods

    def __str__(self) -> str:
        """Return a pretty-print representation."""

        def lines_list_to_str(
                lines_list: List[str],
                prefix: str = '',
                suffix: str = '',
                indent_: int = 0,
                force_multiline: bool = False
        ) -> str:
            """Convert a list of lines into a string.

            Args:
                lines_list: the list of lines to be converted.
                prefix: an optional prefix to be appended at the beginning of
                    the string.
                suffix: an optional suffix to be appended at the end of the string.
                indent_: the optional number of spaces to indent the code.
                force_multiline: force the output to be multiline.

            Returns:
                The lines collapsed into a single string (potentially with line
                breaks).
            """
            if force_multiline or len(lines_list) > 3 or any(
                    '\n' in line for line in lines_list):
                # Return a multi-line string.
                lines_str = indent(',\n'.join(lines_list), ' '*indent_)
                prefix = '{}\n'.format(prefix) if prefix else prefix
                suffix = '\n{}'.format(suffix) if suffix else suffix
            else:
                # Return an inline string.
                lines_str = ', '.join(lines_list)

            return '{}{}{}'.format(prefix, lines_str, suffix)

        def field_to_str(field_value: Any) -> str:
            """Return a string representation of the value of a field.

            Args:
                field_value: the object that contains a field value.

            Returns:
                The string representation of the field (potentially with line
                breaks).
            """
            field_lines = []
            force_multiline = False

            # Handle special cases.
            if isinstance(field_value, list) and len(value) > 0:
                # For non-emtpy lists, always use multiline, with one item per line.
                for item in field_value:
                    field_lines.append(indent('{}'.format(str(item)), ' '*4))
                force_multiline = True
            else:
                field_lines.append(str(field_value))

            prefix = '[' if force_multiline else ''
            suffix = ']' if force_multiline else ''
            return lines_list_to_str(
                field_lines, prefix, suffix, force_multiline=force_multiline)

        def is_skippable(field: Field, value: Any) -> bool:
            """Return whether a field should be skipped."""
            if value == field.default:
                # Skip fields with the default value.
                return True

            if 'hide_if' in field.metadata and field.metadata.get('hide_if') == value:
                return True

            return False

        # Main loop.

        # Build the list of lines.
        fields_lines = []
        for field in fields(self):
            value = getattr(self, field.name)

            # Exclude fields.
            if is_skippable(field, value):
                continue

            # Convert the value into a string, falling back to repr if needed.
            try:
                value_str = field_to_str(value)
            except Exception:  # pylint: disable=broad-except
                value_str = str(value)

            fields_lines.append('{}={}'.format(field.name, value_str))

        # Convert the full object to str.
        output = lines_list_to_str(
            fields_lines, '{}('.format(self.__class__.__name__), ')', 4)

        return output
