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

"""aihwkit example: Example of implementing GPT-2 based model distilgpt2 on the wikitext-2-raw-v1
dataset using AIHWKit. The example demonstrates how to convert the model to analog, run fine-tuning,
text-generation, and inference.

This example was initiated by Gyujun Jeong (gjeong35@gatech.edu).

Use command-line arguments:
For text generation (of both digital and analog models), use arguments: "gt", "L", "c", "pt"
For digital model fine-tuning and loss calculation, use arguments: "d", "c", "lr", "L"
For analog model HWA fine-tuning and loss calculation, use arguments: "t", "c", "n", "lr", "L"

**Source**:
    The example is adapted from code in
    https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb
"""
# pylint: disable=import-error, too-many-arguments, invalid-name


# revise original script: correct digital model code; add text generation code

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

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from datasets import load_dataset, DatasetDict
import datasets.exceptions
import numpy as np

from aihwkit.simulator.configs import (
    TorchInferenceRPUConfig,
    WeightModifierType,
    WeightClipType,
    # WeightNoiseType,
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

# Parse some arguments
PARSER = ArgumentParser("Analog GPT-2 on wikitext-2-raw-v1 example")
PARSER.add_argument("-d", "--digital", help="Add to use digital inference", action="store_true")
PARSER.add_argument(
    "-i",
    "--ideal",
    help="Use ideal config instead of default noisy one",
    action="store_true")
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
    "-L",
    "--load",
    help="Use when loading from training checkpoint",
    action="store_true")
PARSER.add_argument("-c", "--checkpoint", help="Checkpoint file name and path", type=str)
PARSER.add_argument(
    "-lr",
    "--learning_rate",
    help="Learning rate for training",
    default=2e-4, type=float)
PARSER.add_argument("-gt", "--gen_txt", help="Generate text (Inference)", action="store_true")
PARSER.add_argument(
    "-pt",
    "--prompt",
    help="The prompt for text generation",
    default="Once upon a time", type=str)
ARGS = PARSER.parse_args()

# GPT-2 model from Hugging Face model hub
MODEL_NAME = "distilgpt2"  # Smallest GPT-2 model
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
# Set the padding token to eos_token
TOKENIZER.pad_token = TOKENIZER.eos_token

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


def create_ideal_rpu_config(tile_size: int = 512) -> TorchInferenceRPUConfig:
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


def create_rpu_config(modifier_noise, tile_size=512,  # type: ignore[no-untyped-def]
                      dac_res=256, adc_res=256):  # type: ignore[no-untyped-def]
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


def create_model(rpu_config, is_digital) -> AutoModelForCausalLM:   # type: ignore[no-untyped-def]
    """Return Causal Language Model and whether or not it was loaded from a checkpoint"""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # Update model config to use the new pad_token_id
    model.config.pad_token_id = TOKENIZER.pad_token_id
    if not is_digital:
        model = convert_to_analog(model, rpu_config)
        model.remap_analog_weights()
    print(model)
    return model


def preprocess_function(examples) -> AutoTokenizer:   # type: ignore[no-untyped-def]
    """Preprocess the dataset"""
    return TOKENIZER(examples["text"], truncation=True, padding="max_length", max_length=128)


def create_datasets() -> DatasetDict:
    """ Create dataset """
    try:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        # dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        # dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        print("Dataset loaded successfully.")
    except datasets.exceptions.DatasetNotFoundError as err:
        print(f"Error loading dataset: {err}")
        return None, None
    try:
        tokenized_datasets = dataset.map(
            preprocess_function, batched=True,
            remove_columns=["text"], num_proc=4)
        print("Dataset tokenized successfully.")
    except datasets.exceptions.DatasetNotFoundError as err:
        print(f"Error tokenizing dataset: {err}")
        return None, None
    return tokenized_datasets["train"], tokenized_datasets["validation"]


def create_optimizer(model, is_digital):   # type: ignore[no-untyped-def]
    """ Create the optimizer """
    if is_digital:
        optimizer = Adam(model.parameters(), lr=ARGS.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    else:
        # Create the analog-aware optimizer
        optimizer = AnalogSGD(model.parameters(), lr=ARGS.learning_rate)
        optimizer.regroup_param_groups(model)
    return optimizer


def make_trainer(model, optimizer, train_dataset, eval_dataset):  # type: ignore[no-untyped-def]
    """Create the Huggingface Trainer"""
    training_args = TrainingArguments(
        output_dir="./",
        save_strategy="no",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.0001,
        no_cuda=False,
        logging_steps=500,
        eval_strategy="steps",
        eval_steps=500,
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


def do_inference(model, trainer, writer, is_digital,  # type: ignore[no-untyped-def]
                 *, max_inference_time=1e6, n_times=9):  # type: ignore[no-untyped-def]
    """Perform inference experiment at weight noise level specified at runtime."""
    def predict() -> float:
        # Perform inference + evaluate metric here
        result = trainer.evaluate()
        return result["eval_loss"]

    def write_metrics(eval_loss, t_inference) -> None:   # type: ignore[no-untyped-def]
        # Add information to tensorboard
        writer.add_scalar("val/loss", eval_loss, t_inference)
        if ARGS.wandb:
            wandb.log({"t_inference": t_inference, "loss": eval_loss})
        print(f"Inference_time: {t_inference: .2e}" f"Loss: {eval_loss: .2f}\t")

    model.eval()
    # Get the initial metrics
    eval_loss = predict()
    write_metrics(eval_loss, 0.0)
    if not is_digital:
        t_inference_list = np.logspace(0, np.log10(float(max_inference_time)), n_times).tolist()
        for t_inference in t_inference_list:
            model.drift_analog_weights(t_inference)
            eval_loss = predict()
            write_metrics(eval_loss, t_inference)


def main() -> None:
    """Provide the lambda function for WandB sweep. If WandB is not used, then this
    is what is executed in the job
    """
    if ARGS.wandb:
        wandb.init()
    # Define RPU configuration and use it to create model and tokenizer
    if ARGS.ideal:
        rpu_config = create_ideal_rpu_config()
    else:
        rpu_config = create_rpu_config(modifier_noise=ARGS.noise)  # type: ignore[no-untyped-call]
    model = create_model(rpu_config, ARGS.digital)
    # Load dataset
    train_dataset, eval_dataset = create_datasets()
    if train_dataset is None or eval_dataset is None:
        print("Error: train_dataset or eval_dataset is None.")
        return
    # Create optimizer and trainer
    optimizer = create_optimizer(model, ARGS.digital)     # type: ignore[no-untyped-call]
    trainer, writer = make_trainer(model, optimizer,  # type: ignore[no-untyped-call]
                                   train_dataset, eval_dataset)  # type: ignore[no-untyped-call]
    # If "-L", load checkpoint file
    if ARGS.load and ARGS.checkpoint is not None:
        print(f"Load model from '{ARGS.checkpoint}'.")
        model.load_state_dict(torch.load(ARGS.checkpoint, weights_only=False), strict=False)
    # Finetune digital or analog model
    if (ARGS.train_hwa or ARGS.digital) and not ARGS.load:
        trainer.train()
        torch.save(model.state_dict(), ARGS.checkpoint)
    # Calculate inference loss
    do_inference(model, trainer, writer, ARGS.digital)  # type: ignore[no-untyped-call]


def generate_text(prompt, model, max_length=50) -> str:   # type: ignore[no-untyped-def]
    """ Generate a text from the model using a prompt """
    device = next(model.parameters()).device
    encoding = TOKENIZER(prompt, return_tensors='pt')
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        repetition_penalty=5.0,
        pad_token_id=TOKENIZER.eos_token_id,
    )
    text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return text


if ARGS.wandb:
    wandb.agent(SWEEP_ID, function=main, count=4)
elif ARGS.gen_txt:
    if ARGS.ideal:
        rpu_config_defined = create_ideal_rpu_config()
    else:
        rpu_config_defined = create_rpu_config(modifier_noise=ARGS.noise) # type: ignore[no-untyped-call]

    Model = create_model(rpu_config_defined, ARGS.digital)

    if ARGS.load:
        print(f"Loading weights from {ARGS.checkpoint}")
        Model.load_state_dict(torch.load(ARGS.checkpoint, weights_only=False), strict=False)

    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model.to(Device)
    Model.eval()
    print(generate_text(ARGS.prompt, Model))
else:
    main()
