# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""aihwkit example: Example using convert_to_analog to run GPT-2 transformer
**Source**:
    The example is adapted from code in
    https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb
"""
# pylint: disable=invalid-name, too-many-locals, import-error

from datetime import datetime
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from numpy import log10, logspace, argsort
from transformers.integrations import TensorBoardCallback
from transformers import DataCollatorForLanguageModeling

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
)

from torch import save as torch_save, load as torch_load
from torch.utils.tensorboard import SummaryWriter

from evaluate import load
from datasets import load_dataset

from aihwkit.simulator.configs import (
    TorchInferenceRPUConfig,
    InferenceRPUConfig,
    WeightModifierType,
    WeightClipType,
    WeightNoiseType,
    BoundManagementType,
    NoiseManagementType,
    WeightClipParameter,
    WeightModifierParameter,
    MappingParameter,
)

from aihwkit.simulator.presets import PresetIOParameters
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD


# GPT-2 model from Hugging Face model hub
MODEL_NAME = "distilbert/distilgpt2" # smallest model
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add padding token if it doesn't exist
if TOKENIZER.pad_token is None:
    TOKENIZER.add_special_tokens({'pad_token': TOKENIZER.eos_token})

# Parse some arguments
PARSER = ArgumentParser("Analog GPT-2 example")
PARSER.add_argument("-d", "--digital", help="Add to use digital inference", action="store_true")
PARSER.add_argument(
    "-i",
    "--ideal",
    help="Add to use ideal config instead of default noisy one",
    action="store_true",
)
PARSER.add_argument("-w", "--wandb", help="Add to use wandb", action="store_true")
PARSER.add_argument("-n", "--noise", help="Modifier noise", default=0.1, type=float)
PARSER.add_argument(
    "-r",
    "--run_name",
    help="Tensorboard run name",
    default=datetime.now().strftime("%Y%m%d-%H%M%S"),
    type=str,
)
PARSER.add_argument("-t", "--train_hwa", help="Use Hardware-Aware training", action="store_true")
PARSER.add_argument(
    "-L", "--load", help="Use when loading from training checkpoint", action="store_true"
)

PARSER.add_argument(
    "-c",
    "--checkpoint",
    help="File name specifying where to load/save a checkpoint",
    default="./saved_chkpt.pth",
    type=str,
)
PARSER.add_argument(
    "-l", "--learning_rate", help="Learning rate for training", default=2e-4, type=float
)

ARGS = PARSER.parse_args()

if ARGS.wandb:
    import wandb

    # Define weights noise sweep configuration
    SWEEP_CONFIG = {
        "method": "random",
        "name": "modifier noise sweep",
        "metric": {"goal": "maximize", "name": "exact_match"},
        "parameters": {"modifier_noise": {"values": [0, 0.05, 0.1, 0.2]}},
    }

    SWEEP_ID = wandb.sweep(sweep=SWEEP_CONFIG, project="gpt2-weight-noise-experiment")

def create_ideal_rpu_config(tile_size=512):
    """Create RPU Config with ideal conditions"""
    rpu_config = TorchInferenceRPUConfig(
        mapping=MappingParameter(
            digital_bias=True,
            learn_out_scaling=True,
            weight_scaling_omega=1.0,
            out_scaling_columnwise=False,
            weight_scaling_columnwise=True,
            max_input_size=tile_size,
            max_output_size=0,
        ),
        forward=PresetIOParameters(is_perfect=True),
        noise_model=PCMLikeNoiseModel(prog_noise_scale=0.0, read_noise_scale=0.0, drift_scale=0.0),
        drift_compensation=None,
    )
    return rpu_config


def create_rpu_config(modifier_noise, tile_size=512, dac_res=256, adc_res=256):
    """Create RPU Config emulated typical PCM Device"""
    if ARGS.wandb:
        modifier_noise = wandb.config.modifier_noise

    rpu_config = InferenceRPUConfig(
        clip=WeightClipParameter(type=WeightClipType.FIXED_VALUE, fixed_value=1.0),
        modifier=WeightModifierParameter(
            rel_to_actual_wmax=True, type=WeightModifierType.ADD_NORMAL, std_dev=modifier_noise
        ),
        mapping=MappingParameter(
            digital_bias=True,
            learn_out_scaling=True,
            weight_scaling_omega=1.0,
            out_scaling_columnwise=True,
            weight_scaling_columnwise=True,
            max_input_size=tile_size,
            max_output_size=0,
        ),
        forward=PresetIOParameters(
            w_noise_type=WeightNoiseType.PCM_READ,
            w_noise=0.0175,
            inp_res=dac_res,
            out_res=adc_res,
            out_bound=10.0,
            out_noise=0.04,
            bound_management=BoundManagementType.ITERATIVE,
            noise_management=NoiseManagementType.ABS_MAX,
        ),
        noise_model=PCMLikeNoiseModel(),
        drift_compensation=GlobalDriftCompensation(),
    )
    return rpu_config


def create_model(rpu_config):
    """Return GPT-2 model and whether or not it was loaded from a checkpoint"""

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if not ARGS.digital:
        model = convert_to_analog(model, rpu_config)
        model.remap_analog_weights()

    print(model)
    return model

# Preprocess the dataset for GPT-2
def preprocess_data(examples):
    return TOKENIZER(examples["text"], truncation=True, max_length=512, padding="max_length")

# Create datasets
def create_datasets():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=["text"])
    # Set the format for PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return tokenized_dataset

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
        weight_decay=0.001,
        no_cuda=False,
    )

    collator = DefaultDataCollator()

    log_dir = "logs/fit/" + ARGS.run_name
    writer = SummaryWriter(log_dir=log_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_data,
        eval_dataset=tokenized_data,
        tokenizer=TOKENIZER,
        optimizers=(optimizer, None),
        callbacks=[TensorBoardCallback(writer)],
    )

    return trainer, writer


def do_inference(model, trainer, dataset, writer, max_inference_time=1e6, n_times=9):
    """Perform inference experiment at weight noise level specified at runtime."""
    def predict():
        # Perform inference + evaluate metric here
        raw_predictions = trainer.predict(dataset)
        return raw_predictions

    def write_metrics(predictions, t_inference):
        # Add information to tensorboard
        writer.add_scalar("val/predictions", predictions, t_inference)
        if ARGS.wandb:
            wandb.log({"t_inference": t_inference, "predictions": predictions})

        print(f"Predictions: {predictions}\t Drift: {t_inference: .2e}")

    model.eval()

    t_inference_list = logspace(0, log10(float(max_inference_time)), n_times).tolist()

    # Get the initial metrics
    predictions = predict()
    write_metrics(predictions, 0.0)

    for t_inference in t_inference_list:
        model.drift_analog_weights(t_inference)
        predictions = predict()
        write_metrics(predictions, t_inference)

# Main function
def main():
    """Provide the lambda function for WandB sweep. If WandB is not used, then this
    is what is executed in the job
    """
    if ARGS.wandb:
        wandb.init()

    # Define RPU configuration and use it to create model and tokenizer
    if ARGS.ideal:
        rpu_config = create_ideal_rpu_config()
    else:
        rpu_config = create_rpu_config(modifier_noise=ARGS.noise)

    model = create_model(rpu_config)

    tokenized_data = create_datasets()
    optimizer = create_optimizer(model)

    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=TOKENIZER, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./",
        save_strategy="no",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.001,
        no_cuda=False,
    )

    log_dir = "logs/fit/" + ARGS.run_name
    writer = SummaryWriter(log_dir=log_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_data,
        eval_dataset=tokenized_data,
        tokenizer=TOKENIZER,
        optimizers=(optimizer, None),
        callbacks=[TensorBoardCallback(writer)],
    )

    if ARGS.load:
        print(f"Load model from '{ARGS.checkpoint}'.")
        model.load_state_dict(torch_load(ARGS.checkpoint))

    # Do hw-aware training if in analog domain and the model isn't loaded from
    # an existing checkpoint
    if ARGS.train_hwa and not ARGS.digital and not ARGS.load:
        trainer.train()
        torch_save(model.state_dict(), ARGS.checkpoint)

    do_inference(model, trainer, tokenized_data, writer)

# Check if ARGS.wandb is used
if ARGS.wandb:
    wandb.agent(SWEEP_ID, function=main, count=4)
else:
    main()
