# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Basic training Experiment."""

from typing import Any, Dict, List, Tuple, Type, Optional

from torch import device as torch_device, max as torch_max, Tensor
from torch.nn import Module, NLLLoss
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import FashionMNIST, SVHN
from torchvision.transforms import Compose, Normalize, ToTensor

from aihwkit.experiments.experiments.base import Experiment, Signals
from aihwkit.optim import AnalogSGD


class BasicTraining(Experiment):
    """Experiment for training a neural network.

    ``Experiment`` that represents training a neural network using a basic
    training loop.

    This class contains:

    * the data needed for an experiment. The recommended way of setting this
      data is via the arguments of the constructor. Additionally, some of the
      items have getters that are used by the ``Workers`` that execute the
      experiments and by the training loop.
    * the training algorithm, with the main entry point being ``train()``.

    Note:
        When executing a ``BasicTraining`` in the cloud, additional constraints
        are applied to the data. For example, the model is restricted to
        sequential layers of specific types; the dataset choices are limited,
        etc. Please check the ``CloudRunner`` documentation.
    """

    def __init__(
        self,
        dataset: Type[Dataset],
        model: Module,
        batch_size: int = 64,
        loss_function: type = NLLLoss,
        epochs: int = 30,
        learning_rate: float = 0.05,
    ):
        """Create a new ``BasicTraining``.

        Args:
            dataset: the dataset class to be used.
            model: the neural network to be trained.
            batch_size: the batch size used for training.
            loss_function: the loss function used for training.
            epochs: the number of epochs for the training.
            learning_rate: the learning rate used by the optimizer.
        """
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.epochs = epochs
        self.learning_rate = learning_rate

        super().__init__()

    def get_dataset_arguments(self, dataset: type) -> Tuple[Dict, Dict]:
        """Return the dataset constructor arguments for specifying subset."""
        if dataset in (SVHN,):
            return {"split": "train"}, {"split": "test"}
        return {"train": True}, {"train": False}

    def get_dataset_transform(self, dataset: type) -> Any:
        """Return the dataset transform."""
        # Normalize supported datasets.
        if dataset == FashionMNIST:
            mean = Tensor([0.2860])
            std = Tensor([0.3205])
            transform = Compose([ToTensor(), Normalize(mean, std)])
        elif dataset == SVHN:
            mean = Tensor([0.4377, 0.4438, 0.4728])
            std = Tensor([0.1980, 0.2010, 0.1970])
            transform = Compose([ToTensor(), Normalize(mean, std)])
        else:
            transform = Compose([ToTensor()])

        return transform

    def get_data_loaders(
        self,
        dataset: type,
        batch_size: int,
        max_elements_train: int = 0,
        dataset_root: str = "/tmp/datasets",
    ) -> Tuple[DataLoader, DataLoader]:
        """Return `DataLoaders` for the selected dataset.

        Args:
            dataset: the dataset class to be used.
            batch_size: the batch size used for training.
            max_elements_train: the maximum number of elements of the dataset
                to be used. If ``0``, the full dataset is used.
            dataset_root: the path to the folder where the files from the
                dataset are stored.

        Returns:
            A tuple with the training and validation loaders.
        """
        # Create the sets and the loaders.
        train_args, test_args = self.get_dataset_arguments(dataset)
        transform = self.get_dataset_transform(dataset)

        # Create the datasets.
        training_set = dataset(dataset_root, transform=transform, **train_args)
        validation_set = dataset(dataset_root, transform=transform, **test_args)

        if max_elements_train:
            training_set = Subset(training_set, range(max_elements_train))
            validation_set = Subset(validation_set, range(max_elements_train))

        training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

        return training_loader, validation_loader

    def get_optimizer(self, learning_rate: float, model: Module) -> Optimizer:
        """Return the `Optimizer` for the experiment.

        Args:
            learning_rate: the learning rate used by the optimizer.
            model: the neural network to be trained.

        Returns:
            the optimizer to be used in the experiment.
        """
        optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
        optimizer.regroup_param_groups(model)

        return optimizer

    def training_step(
        self,
        training_loader: DataLoader,
        model: Module,
        optimizer: Optimizer,
        loss_function: _Loss,
        device: torch_device,
    ) -> None:
        """Run a single training step.

        Args:
            training_loader: the data loader for the training data.
            model: the neural network to be trained.
            optimizer: the optimizer used for the training.
            loss_function: the loss function used for training.
            device: the torch device used for the model.
        """
        model.train()

        for images, labels in training_loader:
            # Move the data to the device if needed.
            if device:
                images = images.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            batch_image_count = labels.size(0)

            # Run training (backward propagation).
            loss.backward()

            # Optimize weights.
            optimizer.step()

            self._call_hook(
                Signals.TRAIN_EPOCH_BATCH_END, batch_image_count, loss.item() * batch_image_count
            )

    def validation_step(
        self,
        validation_loader: DataLoader,
        model: Module,
        loss_function: _Loss,
        device: torch_device,
    ) -> None:
        """Run a single evaluation step.

        Args:
            validation_loader: the data loader for the validation data.
            model: the neural network to be trained.
            loss_function: the loss function used for training.
            device: the torch device used for the model.
        """
        model.eval()

        for images, labels in validation_loader:
            # Move the data to the device if needed.
            if device:
                images = images.to(device)
                labels = labels.to(device)

            # Predict image.
            prediction = model(images)
            loss = loss_function(prediction, labels)

            _, predicted = torch_max(prediction.data, 1)
            batch_image_count = labels.size(0)
            batch_correct_count = (predicted == labels).sum().item()

            self._call_hook(
                Signals.VALIDATION_EPOCH_BATCH_END,
                batch_image_count,
                batch_correct_count,
                loss.item() * batch_image_count,
            )

    def train(
        self,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        model: Module,
        optimizer: Optimizer,
        loss_function: _Loss,
        epochs: int,
        device: torch_device,
    ) -> List[Dict]:
        """Run the training loop.

        Args:
            training_loader: the data loader for the training data.
            validation_loader: the data loader for the validation data.
            model: the neural network to be trained.
            optimizer: the optimizer used for the training.
            loss_function: the loss function used for training.
            epochs: the number of epochs for the training.
            device: the torch device used for the model.

        Returns:
            A list of the metrics for each epoch.
        """
        results = []

        for epoch_number in range(epochs):
            self._call_hook(Signals.EPOCH_START, epoch_number)
            self._call_hook(Signals.TRAIN_EPOCH_START, epoch_number)
            self.training_step(training_loader, model, optimizer, loss_function, device)
            self._call_hook(Signals.TRAIN_EPOCH_END)

            self._call_hook(Signals.VALIDATION_EPOCH_START, epoch_number)
            self.validation_step(validation_loader, model, loss_function, device)
            self._call_hook(Signals.VALIDATION_EPOCH_END)

            epoch_results = {"epoch": epoch_number}
            epoch_results.update(self._call_hook(Signals.EPOCH_END))
            results.append(epoch_results)

        return results

    def run(
        self,
        max_elements: int = 0,
        dataset_root: str = "/tmp/data",
        device: Optional[torch_device] = None,
    ) -> List[Dict]:
        """Sets up and runs the training.

        Results are returned and the internal model is updated.
        """

        # Build the objects needed for training.
        training_loader, validation_loader = self.get_data_loaders(
            self.dataset,
            self.batch_size,
            max_elements_train=max_elements,
            dataset_root=dataset_root,
        )

        optimizer = self.get_optimizer(self.learning_rate, self.model)

        # Move the model to the device if needed.
        model = self.model
        if device:
            model = model.to(device)

        results = self.train(
            training_loader,
            validation_loader,
            model,
            optimizer,
            self.loss_function(),
            self.epochs,
            device,
        )
        self.model = model  # update the stored model with the trained one
        return results

    def __str__(self) -> str:
        """Return a string representation of a BasicTraining experiment."""
        return (
            "{}(dataset={}, batch_size={}, loss_function={}, epochs={}, "
            "learning_rate={}, model={})".format(
                self.__class__.__name__,
                getattr(self.dataset, "__name__", self.dataset),
                self.batch_size,
                getattr(self.loss_function, "__name__", self.loss_function),
                self.epochs,
                self.learning_rate,
                self.model,
            )
        )
