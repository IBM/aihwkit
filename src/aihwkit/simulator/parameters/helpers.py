# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Utilities for resistive processing units configurations."""
from sys import version_info
from typing import Any, List, Optional, Type
from dataclasses import Field, fields, is_dataclass
from enum import Enum
from textwrap import indent

from aihwkit.simulator import rpu_base
from aihwkit.exceptions import ConfigError
from .enums import RPUDataType

if version_info[0] >= 3 and version_info[1] > 7:
    # pylint: disable=no-name-in-module, ungrouped-imports
    from typing import get_origin  # type: ignore

    HAS_ORIGIN = True
else:
    HAS_ORIGIN = False

ALL_SKIP_FIELD = "is_perfect"
FIELD_MAP = {"forward": "forward_io", "backward": "backward_io"}
ALWAYS_INCLUDE = ["forward", "backward", "update"]


def get_bindings_class(params: Any, data_type: RPUDataType) -> Optional[Type]:
    """Return the data class from the param binding fields.

    Args:
        params: parameter dataclass
        data_type: RPUDataType to use

    Returns:
        the C++ binding class

    Raises:
        ConfigError: if the class is not found
    """
    if getattr(params, "bindings_class", None) is None:
        return None
    if not isinstance(params.bindings_class, str):
        return params.bindings_class
    # string / typed
    class_name = params.bindings_class
    module = getattr(rpu_base, getattr(params, "bindings_module", "devices"))
    if data_type != RPUDataType.FLOAT:
        if not hasattr(module, data_type.value):
            raise ConfigError(
                f"Cannot find requested data_type '{data_type.value}' in rpu_base module. "
            )
        module = getattr(module, data_type.value)
    param_class = getattr(module, class_name, None)
    if param_class is None:
        ConfigError(f"Cannot find requested class '{class_name}' in rpu_base module. ")
    return param_class


def parameters_to_bindings(params: Any, data_type: RPUDataType, check_fields: bool = True) -> Any:
    """Convert a dataclass parameter into a bindings class.

    Args:
        params: parameter dataclass
        data_type: RPUDataType to use
        check_fields: whether to check for the correct attributes

    Returns:
        the C++ bindings

    Raises:
        ConfigError: if the field type mismatches (int to float conversion is ignored)
    """
    # pylint: disable=no-name-in-module, too-many-branches
    result = get_bindings_class(params, data_type)
    if result is None:
        return params
    result = result()

    field_dict = {field.name: (field, getattr(params, field.name)) for field in fields(params)}
    ignore_fields = getattr(params, "bindings_ignore", [])
    if check_fields:
        for key in params.__dict__.keys():
            if key not in field_dict and key not in ignore_fields:
                raise ConfigError(
                    f"Cannot find '{key}' in params "
                    f"'{params.__class__.__name__}'. "
                    "Wrong attribute name?"
                )

    for field, (dataclass_field, value) in field_dict.items():
        # Convert enums to the bindings enums.
        if field in ("unit_cell_devices", "device") or field in ignore_fields:
            # Exclude special fields that are not present in the bindings.
            continue

        if isinstance(value, Enum):
            if hasattr(rpu_base.tiles, value.__class__.__name__):
                enum_class = getattr(rpu_base.tiles, value.__class__.__name__)
            else:
                enum_class = getattr(rpu_base.devices, value.__class__.__name__)
            enum_value = getattr(enum_class, value.value)
            setattr(result, field, enum_value)
        elif is_dataclass(value):
            if hasattr(value, "bindings_class"):
                setattr(result, field, parameters_to_bindings(value, data_type=data_type))
        else:
            if HAS_ORIGIN:
                expected_type = get_origin(dataclass_field.type) or dataclass_field.type
                if (not isinstance(value, expected_type)) and not (
                    expected_type == float
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                ):
                    raise ConfigError(f"Expected type {expected_type} for field {field}")

            setattr(result, field, value)

    return result


def tile_parameters_to_bindings(params: Any, data_type: RPUDataType) -> Any:
    """Convert a tile dataclass parameter into a bindings class.

    Ignores fields that do not have metadata with ``bindings_include`` key.

    Args:
        params: parameter dataclass
        data_type: RPUDataType to use

    Returns:
        the C++ bindings

    """

    result = get_bindings_class(params, data_type)
    if result is None:
        return params

    result = result()  # instantiate results class

    for field in fields(params):
        # Get the mapped field name, if needed.

        if field.name not in ALWAYS_INCLUDE and not field.metadata.get("bindings_include", False):
            continue

        value = params.__dict__[field.name]
        field_name = FIELD_MAP.get(field.name, field.name)

        if isinstance(value, Enum):
            if hasattr(rpu_base.tiles, value.__class__.__name__):
                enum_class = getattr(rpu_base.tiles, value.__class__.__name__)
            else:
                enum_class = getattr(rpu_base.devices, value.__class__.__name__)
            enum_value = getattr(enum_class, value.value)
            setattr(result, field_name, enum_value)

        elif is_dataclass(value):
            if getattr(value, "bindings_class", None) is not None:
                setattr(result, field_name, parameters_to_bindings(value, data_type=data_type))
        else:
            setattr(result, field_name, value)

    return result


class _PrintableMixin:
    """Helper class for pretty-printing of config dataclasses."""

    # pylint: disable=too-few-public-methods

    def __str__(self) -> str:
        """Return a pretty-print representation."""

        def lines_list_to_str(
            lines_list: List[str],
            prefix: str = "",
            suffix: str = "",
            indent_: int = 0,
            force_multiline: bool = False,
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
            if force_multiline or len(lines_list) > 3 or any("\n" in line for line in lines_list):
                # Return a multi-line string.
                lines_str = indent(",\n".join(lines_list), " " * indent_)
                prefix = "{}\n".format(prefix) if prefix else prefix
                suffix = "\n{}".format(suffix) if suffix else suffix
            else:
                # Return an inline string.
                lines_str = ", ".join(lines_list)

            return "{}{}{}".format(prefix, lines_str, suffix)

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
                # For non-empty lists, always use multiline, with one item per line.
                for item in field_value:
                    field_lines.append(indent("{}".format(str(item)), " " * 4))
                force_multiline = True
            else:
                field_lines.append(str(field_value))

            prefix = "[" if force_multiline else ""
            suffix = "]" if force_multiline else ""
            return lines_list_to_str(field_lines, prefix, suffix, force_multiline=force_multiline)

        def is_skippable(field: Field, value: Any) -> bool:
            """Return whether a field should be skipped."""
            if field.metadata.get("always_show", False):
                return False

            if value == field.default:
                # Skip fields with the default value.
                return True

            if "hide_if" in field.metadata and field.metadata.get("hide_if") == value:
                return True

            return False

        # Main loop.

        # Build the list of lines.
        fields_lines = []

        # special case for global skip:
        all_skip = hasattr(self, ALL_SKIP_FIELD) and getattr(self, ALL_SKIP_FIELD)

        for field in fields(self):  # type: ignore[arg-type]
            value = getattr(self, field.name)

            # Exclude fields.
            if (all_skip and field.name != ALL_SKIP_FIELD) or is_skippable(field, value):
                continue

            # Convert the value into a string, falling back to repr if needed.
            try:
                value_str = field_to_str(value)
            except Exception:  # pylint: disable=broad-except
                value_str = str(value)

            fields_lines.append("{}={}".format(field.name, value_str))

        # Convert the full object to str.
        output = lines_list_to_str(fields_lines, "{}(".format(self.__class__.__name__), ")", 4)

        return output
