# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 22: 2-layers LSTM

War and Peace dataset on a 2-layers LSTM inspired network based on the paper:
https://www.frontiersin.org/articles/10.3389/fnins.2018.00745/full
"""

# pylint: disable=redefined-outer-name, too-many-locals, invalid-name, too-many-statements
# pylint: disable=not-callable

# Imports from PyTorch.
import os
import argparse
import time

from typing import Tuple
from torch import tensor, device, FloatTensor, Tensor, transpose, save, load
from torch.nn import CrossEntropyLoss, Linear
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn.functional import one_hot

import numpy as np

from aihwkit.nn import AnalogSequential, AnalogRNN, AnalogLinear, AnalogLSTMCellCombinedWeight
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    build_config,
    SingleRPUConfig,
    SoftBoundsDevice,
    SoftBoundsReferenceDevice,
    ConstantStepDevice,
    MappingParameter,
    UpdateParameters,
)
from aihwkit.simulator.rpu_base import cuda

# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = device("cuda" if USE_CUDA else "cpu")

HIDDEN_DIM = 64
P_DROP = 0.2
TEST_FREQ = 1

# needs to be downloaded, e.g. from Gutenberg
WP_TRAIN_FNAME = "wp.train.txt"
WP_TEST_FNAME = "wp.test.txt"
WP_VALID_FNAME = "wp.valid.txt"
DATASET_PATH = os.path.join(os.getenv("RPUSIM", ""), "data", "lstm")


def parse_args():
    """Parse arguments for the experiment."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--sl", type=int, default=100, help="Sequence length")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument(
        "--rpu_conf",
        type=str,
        default="agad",
        help="""Configuration for the rpu_config. FP is native torch
        using the custom LSTM module of the AIHWKIT. TTV2 is the unit
        cell tile for tiki-takaV2 training""",
    )
    parser.add_argument("--file_name", type=str, default=None, help="Training epochs")
    parser.add_argument(
        "--results_path", type=str, default=None, help="Path where the results will be saved"
    )
    return parser.parse_args()


class WarAndPeaceDataset(Dataset):
    """Custom dataset to load the War and Peace train and test dataset"""

    def __init__(self, path, seq_length: int = 1, train: bool = True):
        super().__init__()

        self.seq_length = seq_length

        # Read the text file
        file_path = os.path.join(path, WP_TRAIN_FNAME)
        with open(file_path, "r", encoding="iso-8859-1") as file:
            text = file.read()
        chars = sorted(list(set(text)))

        # char to index and index to char maps
        char_to_ix = {ch: i for i, ch in enumerate(chars)}

        # If train load the training dataset, otherwise load the test dataset but use the
        # vocabulary of the train dataset for the conversion as some characters are missing
        # from the test dataset
        if train:
            print(
                "Loaded train dataset: ",
                file_path,
                "\nTotal character: ",
                len(text),
                "\nTotal vocabulary: ",
                len(chars),
            )
        else:
            file_path = os.path.join(path, WP_TEST_FNAME)
            with open(file_path, "r", encoding="iso-8859-1") as file:
                text = file.read()
            print(
                "Loaded test dataset: ",
                file_path,
                "\nTotal character: ",
                len(text),
                "\nTotal vocabulary: ",
                len(chars),
            )

        # Convert the letter to integers
        self.characters = tensor([char_to_ix[ch] for ch in text[:-1]])
        # The labels get shifted by one since we want to predict the
        # next character
        self.labels = tensor([char_to_ix[ch] for ch in text[1:]])

        # One hot encoding
        self.characters = one_hot(self.characters, len(chars)).type(FloatTensor)

        # Drop the last characters that won't fit in the multiple of the SEQ_LENGTH
        self.characters = self.characters[
            0 : self.characters.size(0) // self.seq_length * self.seq_length, :
        ]
        self.labels = self.labels[0 : self.characters.size(0) // self.seq_length * self.seq_length]
        self.characters = self.characters.view(-1, self.seq_length, len(chars))
        self.labels = self.labels.view(-1, self.seq_length)

    def __len__(self):
        return len(self.characters)

    def __getitem__(self, idx):
        characters = self.characters[idx, :, :]
        labels = self.labels[idx, :]

        return characters, labels


class WarAndPeaceSampler(Sampler):
    """Custom sampler to load the War and Peace dataset"""

    def __init__(self, dataset, batch_size):
        super().__init__(dataset)

        num_sequence = int(len(dataset))
        num_batches = int(num_sequence / batch_size)

        self.idx_list = []
        for i in range(num_batches):
            for j in range(batch_size):
                self.idx_list.append(i + num_batches * j)

    def __iter__(self):
        return iter(self.idx_list)

    def __len__(self):
        return len(self.idx_list)


class AnalogLSTMLayer(AnalogSequential):
    """Create an LSTM network analogous to the LSTM2-64WP based on the paper:
    https://www.frontiersin.org/articles/10.3389/fnins.2018.00745/full
    """

    def __init__(self, seq_length, vocab_size, hidden_dim, batch_size, rpu_config, p_dropout):
        super().__init__()

        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm = AnalogRNN(
            AnalogLSTMCellCombinedWeight,
            self.vocab_size,
            self.hidden_dim,
            num_layers=2,
            rpu_config=rpu_config,
            dropout=p_dropout,
        )
        if rpu_config is not None:
            self.linear = AnalogLinear(self.hidden_dim, self.vocab_size, rpu_config=rpu_config)
        else:
            self.linear = Linear(self.hidden_dim, self.vocab_size)

    def forward(self, x_in, in_states):
        # pylint: disable=arguments-differ

        x_in = transpose(x_in, 0, 1).contiguous()
        out, _ = self.lstm(x_in, in_states)
        out = transpose(out, 0, 1).contiguous()
        out = out.reshape(self.seq_length * self.batch_size, -1)
        out = self.linear(out)

        return out

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Initialize the hidden states."""
        return self.lstm.get_zero_state(batch_size)


def create_rpu_config(config="FP"):
    """Create an rpu_config for the lstm network

    Args:
        config (str): name of the rpu_config to be returned

    Returns:
        rpu_config: rpu_config to be used in the analog layers

    Type:
        rpu_config: rpu_config

    Raises:
        ValueError: In case config is not found
    """

    mapping = MappingParameter(digital_bias=True, max_input_size=0, max_output_size=0)

    if config == "FP":
        rpu_config = None  # native

    elif config == "RPU_Baseline":
        rpu_config = SingleRPUConfig(
            device=SoftBoundsDevice(), update=UpdateParameters(desired_bl=10), mapping=mapping
        )
        adc_bit = 9
        dac_bit = 7
        rpu_config.forward.out_res = 1 / (2**adc_bit - 2)
        rpu_config.forward.inp_res = 1 / (2**dac_bit - 2)
        rpu_config.forward.out_scale = 2 / 0.6
        rpu_config.backward = rpu_config.forward

    elif config == "RPU_Symmetric":
        rpu_config = SingleRPUConfig(
            device=ConstantStepDevice(dw_min=0.00025, up_down_dtod=0),
            update=UpdateParameters(desired_bl=10),
            mapping=mapping,
        )
        adc_bit = 9
        dac_bit = 7
        rpu_config.forward.out_res = 1 / (2**adc_bit - 2)
        rpu_config.forward.inp_res = 1 / (2**dac_bit - 2)
        rpu_config.forward.out_scale = 2 / 0.6
        rpu_config.backward = rpu_config.forward

    else:
        rpu_config = build_config(
            config, device=SoftBoundsReferenceDevice(dw_min=0.1, subtract_symmetry_point=True)
        )
        adc_bit = 9
        dac_bit = 7
        rpu_config.mapping = mapping
        rpu_config.forward.out_res = 1 / (2**adc_bit - 2)
        rpu_config.forward.inp_res = 1 / (2**dac_bit - 2)
        rpu_config.forward.out_scale = 2 / 0.6
        rpu_config.backward = rpu_config.forward

    return rpu_config


def load_dataset(path, seq_length=1, batch_size=1):
    """Load the dataset and reshape it to provide an input
    of shape [SEQ_LENGTH, BATCH_SIZE, VOCABULARY_SIZE]

    Args:
        path (path): dataset path
        seq_length (int): lenght of the sequence
        batch_size (int): batch size

    Returns:
        train_data: data for the training
        test_data: data for the testing

    Type:
        train_data: dataset
        test_data: dataset
    """

    train_set = WarAndPeaceDataset(path, seq_length=seq_length, train=True)
    test_set = WarAndPeaceDataset(path, seq_length=seq_length, train=False)

    train_data = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=WarAndPeaceSampler(train_set, batch_size),
        drop_last=True,
    )
    test_data = DataLoader(
        test_set,
        batch_size=batch_size,
        sampler=WarAndPeaceSampler(test_set, batch_size),
        drop_last=True,
    )

    return train_data, test_data


def main():
    """Train an AnalogLSTM model with the War and Peace dataset."""

    args = parse_args()
    print("Setting Arguments.. : ", args)

    batch_size = args.bs
    seq_length = args.sl
    learning_rate = args.lr
    epochs = args.epochs
    momentum = 0
    rpu_conf = args.rpu_conf

    path_dataset = DATASET_PATH
    results = os.path.join(os.getcwd(), "results", "LSTM")

    # Set the file name
    file_name = (
        "LSTM_"
        + str(args.rpu_conf)
        + "_E"
        + str(args.epochs)
        + "_BS"
        + str(args.bs)
        + "_LR"
        + str(args.lr)
        + "_"
        + time.strftime("%Y%m%d-%H%M%S")
        if args.file_name is None
        else args.file_name
    )

    path_file = os.path.join(results, file_name)
    print("Saving data to: ", path_file)

    # Load datasets.
    train_data, test_data = load_dataset(path_dataset, seq_length, batch_size)

    # Create rpu_config.
    rpu_config = create_rpu_config(config=rpu_conf)

    # Create model.
    model = AnalogLSTMLayer(
        seq_length=seq_length,
        vocab_size=87,
        hidden_dim=HIDDEN_DIM,
        batch_size=batch_size,
        rpu_config=rpu_config,
        p_dropout=P_DROP,
    ).to(DEVICE)
    print(model)
    print("\nPretty-print of RPU non-default settings:\n")
    print(rpu_config)

    epoch_start = 0
    epoch_losses = []

    if args.file_name is not None:
        print("Loading state dict from: ", (args.file_name + ".ckpt"))
        model.load_state_dict(load((path_file + ".ckpt")))

        with open((path_file + ".csv"), "rb") as file:
            epoch_losses = np.loadtxt(file.read, delimiter=",")
        epoch_start = int(epoch_losses[-1, 0])
        epoch_losses = epoch_losses.tolist()

    # Create optimizer and define loss function
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = CrossEntropyLoss()

    for epoch_number in range(epoch_start + 1, epochs):
        in_states = model.init_hidden(batch_size)
        model.train()
        train_total_loss = 0

        for characters, labels in train_data:
            characters = characters.to(DEVICE)
            labels = labels.view(-1).to(DEVICE)
            optimizer.zero_grad()
            prediction = model(characters, in_states)
            loss = criterion(prediction, labels.view(-1))

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_total_loss += loss.item() * characters.size(0)

        test_total_loss = 0
        if epoch_number % TEST_FREQ == 0:
            in_states = model.init_hidden(batch_size)
            model.eval()

            for characters, labels in test_data:
                characters = characters.to(DEVICE)
                labels = labels.view(-1).to(DEVICE)
                prediction = model(characters, in_states)
                loss = criterion(prediction, labels)
                test_total_loss += loss.item() * characters.size(0)

        epoch_losses.append(
            (
                epoch_number,
                train_total_loss / len(train_data.dataset),
                test_total_loss / len(test_data.dataset),
            )
        )

        print(
            "Epoch {} - Train loss: {:.8f} - Test loss: {:.8f}".format(
                epoch_number,
                train_total_loss / len(train_data.dataset),
                test_total_loss / len(test_data.dataset),
            )
        )

        np.savetxt((path_file + ".csv"), epoch_losses, delimiter=",")
        save(model.state_dict(), (path_file + ".ckpt"))

    with open((path_file + ".config"), "w", encoding="utf-8") as f:
        print("==========================", file=f)
        print("Info about all settings:\n", file=f)
        print(rpu_config, file=f)
        print("==========================", file=f)


if __name__ == "__main__":
    # Execute only if run as the entry point into the program
    main()
