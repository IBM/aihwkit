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
**Source**:
    The example is adapted from code in
    https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb
"""
# pylint: disable=invalid-name
# pylint: disable=too-many-locals

from datetime import datetime
import argparse
import collections
import numpy as np
import os

from transformers.integrations import TensorBoardCallback

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
)

from torch.utils.tensorboard import SummaryWriter

from evaluate import load
from datasets import load_dataset

from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    WeightClipType,
    WeightNoiseType,
    BoundManagementType
)

from aihwkit.simulator.presets.utils import PresetIOParameters
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.nn.modules.container import AnalogSequential
from aihwkit.optim import AnalogSGD


# BERT model from Hugging Face model hub fine-tuned on SQuAD v1
MODEL_NAME = "csarron/bert-base-uncased-squad-v1"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

# Parse some arguments
PARSER = argparse.ArgumentParser("Analog BERT on SQuAD example")
PARSER.add_argument("-d", "--digital",
                    help="Add to use digital inference",
                    action="store_true")
PARSER.add_argument("-i", "--ideal",
                    help="Add to use ideal config instead of default noisy one",
                    action="store_true")
PARSER.add_argument("-w", "--wandb",
                    help="Add to use wandb",
                    action="store_true")
PARSER.add_argument("-n", "--noise",
                    help="Weight noise",
                    default=0.0175,
                    type=float)
PARSER.add_argument("-r", "--run_name",
                    help="Tensorboard run name",
                    default=datetime.now().strftime("%Y%m%d-%H%M%S"),
                    type=str)
PARSER.add_argument("-t", "--train_hwa",
                    help="Use Hardware-Aware training",
                    action="store_true")
PARSER.add_argument("-c", "--checkpoint_dir",
                    help="Directory specifying where to load/save a checkpoint",
                    default="./saved",
                    type=str)
PARSER.add_argument("-l", "--learning_rate",
                    help="Learning rate for training",
                    default=2e-4,
                    type=float)

ARGS = PARSER.parse_args()

if ARGS.wandb:
    import wandb

    # Define weights noise sweep configuration
    SWEEP_CONFIG = {
        "method": "random",
        "name": "weight noise sweep",
        "metric": {
            "goal": "maximize",
            "name": "exact_match"
        },
        "parameters": {
            "weight_noise": {"values": [0, 0.00875, 0.0175, 0.035, 0.07]}
        }
    }

    SWEEP_ID = wandb.sweep(sweep=SWEEP_CONFIG, project="bert-weight-noise-experiment")
else:
    os.environ["WANDB_DISABLED"] = "true"

# max length and stride specific to pretrained model
MAX_LENGTH = 320
DOC_STRIDE = 128


def create_ideal_rpu_config(g_max=160, tile_size=256):
    """Create RPU Config with ideal conditions"""
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
    rpu_config.forward.is_perfect = True
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=g_max, prog_noise_scale=0, read_noise_scale=0)
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config


def create_rpu_config(w_noise, g_max=160, tile_size=256, dac_res=256, adc_res=256):
    """Create RPU Config emulated typical PCM Device"""
    if ARGS.wandb:
        w_noise = wandb.config.weight_noise

    rpu_config = InferenceRPUConfig()
    rpu_config.clip.type = WeightClipType.FIXED_VALUE
    rpu_config.clip.fixed_value = 1.0
    rpu_config.modifier.rel_to_actual_wmax = True
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL
    rpu_config.modifier.std_dev = 0.1
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


def create_model(rpu_config):
    """Return Question Answering model and whether or not it was loaded from a checkpoint"""
    is_checkpoint_model = False
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(ARGS.checkpoint_dir)
        is_checkpoint_model = True
    except:
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    if not ARGS.digital:
        model = AnalogSequential(convert_to_analog_mapped(model, rpu_config))

    print(model)

    return model, is_checkpoint_model


# Some examples in the dataset may have contexts that exceed the maximum input length
# We can truncate the context using truncation="only_second"
def preprocess_train(dataset):
    """Preprocess the training dataset"""
    # Some of the questions have lots of whitespace on the left,
    # which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space).
    # So we remove that
    # left whitespace
    dataset["question"] = [q.lstrip() for q in dataset["question"]]

    # Tokenize our dataset with truncation and padding,
    # but keep the overflows using a stride. This results
    # in one example possibly giving several features when a context is long,
    # each of those features having a
    # context that overlaps a bit the context of the previous feature, the stride being the number
    # of overlapping tokens in the overlap.
    tokenized_dataset = TOKENIZER(
        dataset["question"],
        dataset["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to
    # character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_dataset.pop("offset_mapping")

    # Store start and end character positions for answers in context
    tokenized_dataset["start_positions"] = []
    tokenized_dataset["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_dataset["input_ids"][i]
        cls_index = input_ids.index(TOKENIZER.cls_token_id)

        # Grab the sequence corresponding to that example
        # (to know what is the context and what is the question).
        sequence_ids = tokenized_dataset.sequence_ids(i)

        # One example can give several spans, this
        # is the index of the example containing this span of text.
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

            # Detect if the answer is out of the span
            # (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char):
                tokenized_dataset["start_positions"].append(cls_index)
                tokenized_dataset["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and
                # token_end_index to the two ends of the answer.
                # Note: we could go after the last offset
                # if the answer is the last word (edge case).
                while (token_start_index < len(offsets)
                       and offsets[token_start_index][0] <= start_char):
                    token_start_index += 1
                tokenized_dataset["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_dataset["end_positions"].append(token_end_index + 1)

    return tokenized_dataset


def preprocess_validation(dataset):
    """Preprocess the validation set"""
    # Some of the questions have lots of whitespace on the left,
    # which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space).
    # So we remove that
    # left whitespace
    dataset["question"] = [q.lstrip() for q in dataset["question"]]

    # Tokenize our dataset with truncation and maybe padding,
    # but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long,
    # each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_dataset = TOKENIZER(
        dataset["question"],
        dataset["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_dataset["example_id"] = []

    for i in range(len(tokenized_dataset["input_ids"])):
        # Grab the sequence corresponding to that example
        # (to know what is the context and what is the question).
        sequence_ids = tokenized_dataset.sequence_ids(i)
        context_index = 1

        # One example can give several spans,
        # this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_dataset["example_id"].append(dataset["id"][sample_index])

        # Set to None the offset_mapping that are not
        # part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_dataset["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_dataset["offset_mapping"][i])
        ]

    return tokenized_dataset


def postprocess_predictions(
    examples,
    features,
    raw_predictions,
    n_best_size=20,
    max_answer_length=30
):
    """Postprocess raw predictions"""
    features.set_format(type=features.format["type"], columns=list(features.features.keys()))
    all_start_logits, all_end_logits = raw_predictions

    # Map examples ids to index
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}

    # Create dict of lists, mapping example indices with corresponding feature indices
    features_per_example = collections.defaultdict(list)

    for i, feature in enumerate(features):
        # For each example, take example_id, map to corresponding index
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill
    predictions = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions"
          f"split into {len(features)} features.")

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

            # This is what will allow us to map some the positions in our
            # logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1:-n_best_size - 1:-1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size - 1:-1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are
                    # out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue

                    # Don't consider answers with a length
                    # that is either < 0 or > max_answer_length.
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
                            "text": context[start_char:end_char]
                        }
                    )

        # If we have valid answers, choose the best one
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction,
            # we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Choose the best answer as the prediction for the current example
        predictions[example["id"]] = best_answer["text"]

    return predictions


def create_datasets():
    """Load the SQuAD dataset, the tokenized version, and the validation set"""
    squad = load_dataset("squad")

    # Preprocessing changes number of samples, so we need to remove some columns so
    # the data updates properly
    tokenized_data = squad.map(
                        preprocess_train,
                        batched=True,
                        remove_columns=squad["train"].column_names)
    eval_data = squad["validation"].map(
                        preprocess_validation,
                        batched=True,
                        remove_columns=squad["validation"].column_names)

    return squad, tokenized_data, eval_data


def create_optimizer(model):
    """Create the analog-aware optimizer"""
    optimizer = AnalogSGD(model.parameters(), lr=ARGS.learning_rate)

    optimizer.regroup_param_groups(model)

    return optimizer


def make_trainer(model, optimizer, tokenized_data):
    """Create the Huggingface Trainer"""
    training_args = TrainingArguments(
        output_dir="./",
        save_strategy="no",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    collator = DefaultDataCollator()

    log_dir = "logs/fit/" + ARGS.run_name
    writer = SummaryWriter(log_dir=log_dir)

    trainer = Trainer(
        model=model if ARGS.digital else model[0],
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=TOKENIZER,
        optimizers=(optimizer, None),
        callbacks=[TensorBoardCallback(writer)]
    )

    return trainer, writer


def do_inference(model, trainer, squad, eval_data, writer, max_inference_time=1e6, n_times=9):
    """Perform inference experiment at weight noise level specified at runtime.
    SQuAD exact match and f1 metrics are captured in Tensorboard
    """
    # Helper functions
    def predict():
        # Perform inference + evaluate metric here
        raw_predictions = trainer.predict(eval_data)
        predictions = postprocess_predictions(
            squad["validation"],
            eval_data,
            raw_predictions.predictions)

        # Format to list of dicts instead of a large dict
        formatted_preds = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        out_metric = metric.compute(predictions=formatted_preds, references=ground_truth)

        return out_metric["f1"], out_metric["exact_match"]

    def write_metrics(f1, exact_match, t_inference):
        # Add information to tensorboard
        writer.add_scalar("val/f1", f1, t_inference)
        writer.add_scalar("val/exact_match", exact_match, t_inference)

        if ARGS.wandb:
            wandb.log({
                "t_inference": t_inference,
                "f1": f1,
                "exact_match": exact_match,
            })

        print(f"Exact match: {exact_match: .2f}\t"
              f"F1: {f1: .2f}\t"
              f"Drift: {t_inference: .2e}")

    model.eval()

    metric = load("squad")

    ground_truth = [{"id": ex["id"], "answers": ex["answers"]} for ex in squad["validation"]]

    t_inference_list = np.logspace(0, np.log10(float(max_inference_time)), n_times).tolist()

    # Get the initial metrics
    f1, exact_match = predict()
    write_metrics(f1, exact_match, 0.0)

    for t_inference in t_inference_list:
        # Only drift and recalculate metrics if in analog
        if not ARGS.digital:
            model.drift_analog_weights(t_inference)

            f1, exact_match = predict()

        write_metrics(f1, exact_match, t_inference)


def main():
    """Provide the lambda function for WandB sweep. If WandB is not used, then this
    is what is executed in the job
    """
    if ARGS.wandb:
        wandb.init()

    # Define RPU configuration and use it to create model and tokenizer
    rpu_config = create_ideal_rpu_config() if ARGS.ideal else create_rpu_config(w_noise=ARGS.noise)
    model, is_checkpoint_model = create_model(rpu_config)
    squad, tokenized_data, eval_data = create_datasets()
    optimizer = create_optimizer(model)
    trainer, writer = make_trainer(model, optimizer, tokenized_data)

    # Do hw-aware training if in analog domain and the model isn't loaded from
    # an existing checkpoint
    if ARGS.train_hwa and not ARGS.digital and not is_checkpoint_model:
        trainer.train()
        trainer.save_model(ARGS.checkpoint_dir)

    do_inference(model, trainer, squad, eval_data, writer)


if ARGS.wandb:
    wandb.agent(SWEEP_ID, function=main, count=4)
else:
    main()
