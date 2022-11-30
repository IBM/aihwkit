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

"""aihwkit example 24: Example using convert_to_analog to run BERT transformer on SQuAD task
"""
# pylint: disable=invalid-name
# pylint: disable=too-many-locals

from datetime import datetime

import numpy as np
import collections

import argparse

from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightClipType, WeightNoiseType, BoundManagementType
from aihwkit.simulator.presets.utils import PresetIOParameters
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.nn.modules.container import AnalogSequential
from aihwkit.optim import AnalogSGD

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
)

from transformers.integrations import TensorBoardCallback

from torch.utils.tensorboard import SummaryWriter

from evaluate import load
from datasets import load_dataset

# Parse some arguments
parser = argparse.ArgumentParser('Analog BERT on SQuAD example')
parser.add_argument('-d', '--digital',
                    help='Add to use digital inference',
                    action='store_true')
parser.add_argument('-i', '--ideal',
                    help='Add to use ideal config instead of default noisy one',
                    action='store_true')

args = parser.parse_args()

# max length and stride specific to pretrained model
max_length=320
doc_stride=128

def create_ideal_rpu_config(g_max=160, tile_size=256, w_noise=0.0, out_noise=0.0):
    rpu_config = InferenceRPUConfig()
    rpu_config.modifier
    rpu_config.clip.type = WeightClipType.FIXED_VALUE
    rpu_config.clip.fixed_value = 1.0
    rpu_config.modifier.rel_to_actual_wmax = True
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.learn_out_scaling_alpha = True
    rpu_config.mapping.weight_scaling_omega = 1
    rpu_config.mapping.weight_scaling_omega_columnwise = True
    rpu_config.mapping.max_input_size = tile_size
    rpu_config.mapping.max_output_size = 255
    rpu_config.forward = PresetIOParameters()
    rpu_config.forward.w_noise_type = WeightNoiseType.PCM_READ
    rpu_config.forward.w_noise = w_noise
    rpu_config.forward.out_noise = out_noise
    rpu_config.forward.inp_res = -1
    rpu_config.forward.out_res = -1
    rpu_config.forward.bound_management = BoundManagementType.ITERATIVE
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=g_max, prog_noise_scale=0, read_noise_scale=0)
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config

def create_rpu_config(g_max=160, tile_size=256, dac_res=256, adc_res=256, w_noise=0.0175):
    rpu_config = InferenceRPUConfig()
    rpu_config.clip.type = WeightClipType.FIXED_VALUE
    rpu_config.clip.fixed_value = 1.0
    rpu_config.modifier.rel_to_actual_wmax = True
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.learn_out_scaling_alpha = True
    rpu_config.mapping.weight_scaling_omega = 1
    rpu_config.mapping.weight_scaling_omega_columnwise = True
    rpu_config.mapping.max_input_size = tile_size
    rpu_config.mapping.max_output_size = 255
    rpu_config.forward = PresetIOParameters()
    rpu_config.forward.w_noise_type = WeightNoiseType.PCM_READ
    rpu_config.forward.w_noise = w_noise
    rpu_config.forward.inp_res = 1/dac_res
    rpu_config.forward.out_res = 1/adc_res
    rpu_config.forward.bound_management = BoundManagementType.ITERATIVE
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=g_max)
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config

def create_model_and_tokenizer(rpu_config):
    model = AutoModelForQuestionAnswering.from_pretrained("csarron/bert-base-uncased-squad-v1")
    if not args.digital:
        model = AnalogSequential(convert_to_analog_mapped(model, rpu_config))
    print(model)

    tokenizer = AutoTokenizer.from_pretrained("csarron/bert-base-uncased-squad-v1")
    return model, tokenizer

rpu_config = create_ideal_rpu_config() if args.ideal else create_rpu_config()
model, tokenizer = create_model_and_tokenizer(rpu_config)

# Some examples in the dataset may have contexts that exceed the maximum input length
# We can truncate the context using truncation="only_second"
def preprocess_train(dataset):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    dataset["question"] = [ q.lstrip() for q in dataset["question"] ]

    # Tokenize our dataset with truncation and padding, but keep the overflows using a stride. This results
    # in one example possibly giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature, the stride being the number
    # of overlapping tokens in the overlap.
    tokenized_dataset = tokenizer(
        dataset["question"],
        dataset["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_dataset.pop("offset_mapping")

    # Store start and end character positions for answers in context
    tokenized_dataset["start_positions"] = []
    tokenized_dataset["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_dataset["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_dataset.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = dataset["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_dataset["start_positions"].append(cls_index)
            tokenized_dataset["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_dataset["start_positions"].append(cls_index)
                tokenized_dataset["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_dataset["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_dataset["end_positions"].append(token_end_index + 1)

    return tokenized_dataset

def preprocess_validation(dataset):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    dataset["question"] = [ q.lstrip() for q in dataset["question"] ]

    # Tokenize our dataset with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_dataset = tokenizer(
        dataset["question"],
        dataset["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_dataset["example_id"] = []

    for i in range(len(tokenized_dataset["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_dataset.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_dataset["example_id"].append(dataset["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_dataset["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_dataset["offset_mapping"][i])
        ]

    return tokenized_dataset

def postprocess_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    features.set_format(type=features.format["type"], columns=list(features.features.keys()))
    all_start_logits, all_end_logits = raw_predictions

    # Map examples ids to index
    example_id_to_index = { k: i for i, k in enumerate(examples["id"]) }

    # Create dict of lists, mapping example indices with corresponding feature indices
    features_per_example = collections.defaultdict(list)

    for i, feature in enumerate(features):
        # For each example, take example_id, map to corresponding index
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill
    predictions = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Loop over all examples
    for example_index, example in enumerate(examples):
        # Find the feature indices corresponding to the current example
        feature_indices = features_per_example[example_index]

        # Store valid answers
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    # Map the start token to the index of the start of that token in the context
                    # Map the end token to the index of the end of that token in the context
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]

                    # Add the answer
                    # Score is the sum of logits for the start and end position of the answer
                    # Include the text which is taken directly from the context
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char : end_char]
                        }
                    )

        # If we have valid answers, choose the best one
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Choose the best answer as the prediction for the current example
        predictions[example["id"]] = best_answer["text"]

    return predictions

def create_datasets():
    squad = load_dataset("squad")

    # Preprocessing changes number of samples, so we need to remove some columns so
    # the data updates properly
    tokenized_data = squad.map(preprocess_train, batched=True, remove_columns=squad["train"].column_names)
    eval_data = squad["validation"].map(preprocess_validation, batched=True, remove_columns=squad["validation"].column_names)

    return squad, tokenized_data, eval_data

def create_optimizer(model):
    """Create the analog-aware optimizer"""

    optimizer = AnalogSGD(model.parameters(), lr=2e-4)

    optimizer.regroup_param_groups(model)

    return optimizer

def make_trainer(model, optimizer, tokenized_data, tokenizer):
    training_args = TrainingArguments(
        output_dir='./',
        save_strategy="no",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    collator = DefaultDataCollator()

    log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    trainer = Trainer(
        model=model if args.digital else model[0],
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        callbacks=[ TensorBoardCallback(writer) ]
    )

    return trainer, writer

def do_inference(model, trainer, squad, eval_data, writer, max_inference_time=1e6, n_times=9):
    model.eval()

    metric = load("squad")

    ground_truth = [ { "id": ex["id"], "answers": ex["answers"] } for ex in squad["validation"] ]

    t_inference_list = [0.0] + np.logspace(0, np.log10(float(max_inference_time)), n_times).tolist()

    for t_inference in t_inference_list:
        model.drift_analog_weights(t_inference)

        # Perform inference + evaluate metric here
        raw_predictions = trainer.predict(eval_data)
        predictions = postprocess_predictions(squad["validation"], eval_data, raw_predictions.predictions)

        # Format to list of dicts instead of a large dict
        formatted_preds = [ { "id": k, "prediction_text": v } for k, v in predictions.items() ]

        out_metric = metric.compute(predictions=formatted_preds, references=ground_truth)
        f1, exact_match = out_metric["f1"], out_metric["exact_match"]

        writer.add_scalar('val/f1', f1, t_inference)
        writer.add_scalar('val/exact_match', exact_match, t_inference)

        exact_match, f1, drift = exact_match, f1, t_inference
        print(f"Exact match: {exact_match: .2f}\t"
              f"F1: {f1: .2f}\t"
              f"Drift: {drift: .2e}")

squad, tokenized_data, eval_data = create_datasets()
optimizer = create_optimizer(model)
trainer, writer = make_trainer(model, optimizer, tokenized_data, tokenizer)

# Do hw-aware training
# trainer.train()

do_inference(model, trainer, squad, eval_data, writer)


''' Next steps
        - Drift Experiment
            - Take digital model and fine-tune it on a task + use it for inference
            - Perform hardware-aware training with model
            - Show that hwa training creates a more robust model
                since drift is accomodated for over time
            - Integrate wandb for hypterparameter tuning
        - Add setup/installation steps needed to run example to README
        - Distributed compute
        - Convert to notebook
'''
