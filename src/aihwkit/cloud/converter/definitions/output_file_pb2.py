# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: output_file.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import onnx_common_pb2 as onnx__common__pb2
from . import common_pb2 as common__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x11output_file.proto\x12\x05\x61ihwx\x1a\x11onnx_common.proto\x1a\x0c\x63ommon.proto"I\n\x10\x45pochInformation\x12\r\n\x05\x65poch\x18\x01 \x02(\x05\x12&\n\x07metrics\x18\x02 \x03(\x0b\x32\x15.aihwx.AttributeProto"\xa3\x01\n\x0eTrainingOutput\x12\x1f\n\x07version\x18\x01 \x02(\x0b\x32\x0e.aihwx.Version\x12\x1f\n\x07network\x18\x02 \x02(\x0b\x32\x0e.aihwx.Network\x12\'\n\x06\x65pochs\x18\x03 \x03(\x0b\x32\x17.aihwx.EpochInformation\x12&\n\x07metrics\x18\x04 \x03(\x0b\x32\x15.aihwx.AttributeProto'
)

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "output_file_pb2", globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _EPOCHINFORMATION._serialized_start = 61
    _EPOCHINFORMATION._serialized_end = 134
    _TRAININGOUTPUT._serialized_start = 137
    _TRAININGOUTPUT._serialized_end = 300
# @@protoc_insertion_point(module_scope)
