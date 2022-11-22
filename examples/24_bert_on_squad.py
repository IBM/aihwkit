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

from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightClipType, WeightNoiseType, BoundManagementType
from aihwkit.simulator.presets.utils import PresetIOParameters
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog_mapped
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

from datasets import load_dataset, load_metric

def create_rpu_config(g_max = 160, tile_size=256, dac_res=256, adc_res=256, w_noise=0.1):
    rpu_config = InferenceRPUConfig()
    rpu_config.clip.type = WeightClipType.FIXED_VALUE
    rpu_config.clip.fixed_value = 1.0
    rpu_config.modifier.rel_to_actual_wmax = True
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.learn_out_scaling_alpha = True
    rpu_config.mapping.weight_scaling_omega = 0.4
    rpu_config.mapping.weight_scaling_omega_columnwise = True
    rpu_config.mapping.max_input_size = tile_size
    rpu_config.mapping.max_output_size = 255
    rpu_config.forward = PresetIOParameters()
    rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
    rpu_config.forward.w_noise = w_noise
    rpu_config.forward.inp_res = 1/dac_res
    rpu_config.forward.out_res = 1/adc_res
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=g_max)
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config

def create_model_and_tokenizer(rpu_config):
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model = convert_to_analog_mapped(model, rpu_config)

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    return model, tokenizer

# From huggingface tutorial on question answering
# Some examples in the dataset may have contexts that exceed the maximum input length
#   We can truncate the context using truncation="only_second"
def preprocess_train(dataset, tokenizer):
    questions = [ q.strip() for q in dataset["question"] ]
    inputs = tokenizer(
        questions,
        dataset["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length"
    )

    offset_mappings = inputs.pop("offset_mapping")
    answers = dataset["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mappings):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)

        # offset[context_start][0] gets the character position corresponding to the start
        # of the token at the beginning of the context
        # offset[context_start][1] gets the character position corresponding to the end
        # of the token at the end of the context

        # We then check that the answer is contained within these corresponding positions
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            # The answer isn't contained within the context, so add (0, 0)
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions

            # Shift a pointer from the beginning of the context over to the beginning of
            # the first token in the answer
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            # Shift a pointer from the end of the context over to the end of
            # the last token in the answer
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation(dataset, tokenizer):
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
        max_length=384,
        stride=128,
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

def create_dataset():
    squad = load_dataset("squad")

    print(squad["train"][0])

    # Preprocessing changes number of samples, so we need to remove some columns so
    # the data updates properly
    squad = squad.map(preprocess_train, batched=True, remove_columns=squad["train"].column_names)
    return squad

def create_optimizer(model):
    """Create the analog-aware optimizer"""

    optimizer = AnalogSGD(model.parameters(), lr=2e-4)

    optimizer.regroup_param_groups(model)

    return optimizer

def make_trainer(model, optimizer, squad, tokenizer):
    training_args = TrainingArguments(
        output_dir='./',
        save_strategy="no",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    collator = DefaultDataCollator()

    log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=squad["train"],
        eval_dataset=squad["validation"],
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        callbacks=TensorBoardCallback(writer)
    )

    return trainer

def do_inference(model, trainer, squad, max_inference_time=1e6, n_times=9):
    eval_data = squad["validation"].map(preprocess_validation, batched=True, remove_columns=squad["validation"].column_names)

    metric = load_metric("squad")
    drift_metrics = []

    ground_truth = [ { "id": ex["id"], "answers": ex["answers"] } for ex in squad["validation"] ]

    t_inference_list = [0.0] + np.logspace(0, np.log10(float(max_inference_time)), n_times).tolist()

    for t_inference in t_inference_list:
        model.drift_analog_weights(t_inference)

        # Perform inference + evaluate metric here
        raw_predictions = trainer.predict(eval_data)
        predictions = postprocess_predictions(squad["validation"], eval_data, raw_predictions)

        # Format to list of dicts instead of a large dict
        formatted_preds = [ { "id": k, "text": v } for k, v in predictions.items() ]
        drift_metrics.append(metric.compute(predictions=formatted_preds, references=ground_truth))

        exact_match, f1, drift = drift_metrics[-1]["exact_match"], drift_metrics[-1]["f1"], t_inference
        print(f"Exact match: {exact_match: .2f}\t"
              f"F1: {f1: .2f}\t"
              f"Drift: {drift: .2e}")

rpu_config = create_rpu_config()
model, tokenizer = create_model_and_tokenizer(rpu_config)
squad = create_dataset()
optimizer = create_optimizer(model)
trainer = make_trainer(model, optimizer, squad, tokenizer)

do_inference(model, trainer, squad)


''' Next steps
        - Convert to notebook
        - Drift Experiment
            - Take digital model and fine-tune it on a task + use it for inference
            - Perform hardware-aware training with model
            - Show that hwa training creates a more robust model
                since drift is accomodated for over time
'''