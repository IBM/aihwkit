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

"""aihwkit example: Example using convert_to_analog to run GPT-2 transformer on openwebtext
**Source**:
    The example is adapted from code in
    https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb
"""
# pylint: disable=invalid-name, too-many-locals, import-error

from datetime import datetime
from argparse import ArgumentParser
from transformers.integrations import TensorBoardCallback

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from torch import save as torch_save, load as torch_load
from torch.utils.tensorboard import SummaryWriter

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
MODEL_NAME = "distilbert/distilgpt2"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

# Parse some arguments
PARSER = ArgumentParser("Analog GPT-2 on openwebtext example")
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
    "-L", "--load", help="Use when loadiung from training checkpoint", action="store_true"
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
        "metric": {"goal": "maximize", "name": "perplexity"},
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

    rpu_config = TorchInferenceRPUConfig(
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
    """Return Causal Language Model and whether or not it was loaded from a checkpoint"""

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if not ARGS.digital:
        model = convert_to_analog(model, rpu_config)
        model.remap_analog_weights()

    print(model)
    return model


def preprocess_function(examples):
    """Preprocess the dataset"""
    return TOKENIZER(examples["text"], truncation=True, padding="max_length", max_length=128)


def create_datasets():
    """Load the openwebtext dataset"""
    dataset = load_dataset("openwebtext", split="train[:1%]").train_test_split(test_size=0.1)
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
    return tokenized_datasets["train"], tokenized_datasets["test"]


def create_optimizer(model):
    """Create the analog-aware optimizer"""
    optimizer = AnalogSGD(model.parameters(), lr=ARGS.learning_rate)
    optimizer.regroup_param_groups(model)
    return optimizer


def make_trainer(model, optimizer, train_dataset, eval_dataset):
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

    collator = DataCollatorForLanguageModeling(tokenizer=TOKENIZER, mlm=False)

    log_dir = "logs/fit/" + ARGS.run_name
    writer = SummaryWriter(log_dir=log_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=TOKENIZER,
        optimizers=(optimizer, None),
        callbacks=[TensorBoardCallback(writer)],
    )

    return trainer, writer


def do_inference(model, trainer, eval_dataset, writer, max_inference_time=1e6, n_times=9):
    """Perform inference experiment at weight noise level specified at runtime."""
    def predict():
        # Perform inference + evaluate metric here
        result = trainer.evaluate()
        return result["eval_loss"]

    def write_metrics(eval_loss, t_inference):
        # Add information to tensorboard
        writer.add_scalar("val/loss", eval_loss, t_inference)

        if ARGS.wandb:
            wandb.log({"t_inference": t_inference, "loss": eval_loss})

        print(f"Loss: {eval_loss: .2f}\t" f"Drift: {t_inference: .2e}")

    model.eval()

    t_inference_list = logspace(0, log10(float(max_inference_time)), n_times).tolist()

    # Get the initial metrics
    eval_loss = predict()
    write_metrics(eval_loss, 0.0)

    for t_inference in t_inference_list:
        model.drift_analog_weights(t_inference)
        eval_loss = predict()
        write_metrics(eval_loss, t_inference)


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

    train_dataset, eval_dataset = create_datasets()
    optimizer = create_optimizer(model)
    trainer, writer = make_trainer(model, optimizer, train_dataset, eval_dataset)

    if ARGS.load:
        print(f"Load model from '{ARGS.checkpoint}'.")
        model.load_state_dict(torch_load(ARGS.checkpoint))

    # Do hw-aware training if in analog domain and the model isn't loaded from
    # an existing checkpoint
    if ARGS.train_hwa and not ARGS.digital and not ARGS.load:
        trainer.train()
        torch_save(model.state_dict(), ARGS.checkpoint)
    do_inference(model, trainer, eval_dataset, writer)


if ARGS.wandb:
    wandb.agent(SWEEP_ID, function=main, count=4)
else:
    main()
