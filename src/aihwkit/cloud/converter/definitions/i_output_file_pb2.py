# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: i_output_file.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import i_common_pb2 as i__common__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13i_output_file.proto\x12\x11\x61ihwx.inferencing\x1a\x0ei_common.proto\"}\n\x15InferenceResultsProto\x12\x13\n\x0bt_inference\x18\x01 \x02(\x02\x12\x14\n\x0c\x61vg_accuracy\x18\x02 \x02(\x02\x12\x14\n\x0cstd_accuracy\x18\x03 \x02(\x02\x12\x11\n\tavg_error\x18\x04 \x02(\x02\x12\x10\n\x08\x61vg_loss\x18\x05 \x02(\x02\"\x9d\x01\n\x12InferenceRunsProto\x12\x18\n\x10inference_repeat\x18\x01 \x02(\x03\x12\x12\n\nis_partial\x18\x02 \x02(\x08\x12\x14\n\x0ctime_elapsed\x18\x03 \x02(\x02\x12\x43\n\x11inference_results\x18\x04 \x03(\x0b\x32(.aihwx.inferencing.InferenceResultsProto\"\x7f\n\x11InferencingOutput\x12+\n\x07version\x18\x01 \x02(\x0b\x32\x1a.aihwx.inferencing.Version\x12=\n\x0einference_runs\x18\x02 \x02(\x0b\x32%.aihwx.inferencing.InferenceRunsProto')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'i_output_file_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _INFERENCERESULTSPROTO._serialized_start=58
  _INFERENCERESULTSPROTO._serialized_end=183
  _INFERENCERUNSPROTO._serialized_start=186
  _INFERENCERUNSPROTO._serialized_end=343
  _INFERENCINGOUTPUT._serialized_start=345
  _INFERENCINGOUTPUT._serialized_end=472
# @@protoc_insertion_point(module_scope)
