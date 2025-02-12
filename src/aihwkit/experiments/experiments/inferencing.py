# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Basic inferencing Experiment."""

# pylint: disable=too-many-locals

from typing import Any, Dict, Tuple, Type, Optional
from os import path, mkdir
from copy import deepcopy
from requests import get as requests_get
from numpy import ndarray, array, logspace, log10, zeros, concatenate


from torch import device as torch_device, max as torch_max, Tensor
from torch import load
from torch.nn import Module, CrossEntropyLoss
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import FashionMNIST, SVHN
from torchvision.transforms import Compose, Normalize, ToTensor

from aihwkit.experiments.experiments.base import Experiment, Signals
from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.utils.legacy import convert_legacy_checkpoint


WEIGHT_TEMPLATE_URL = "https://github.com/IBM-AI-Hardware-Center/Composer/raw/main/"


def download(url: str, destination: str) -> None:
    """Helper for downloading a file from url"""
    response = requests_get(url, timeout=30.0)
    with open(destination, "wb") as file_:
        file_.write(response.content)


class BasicInferencing(Experiment):
    """Experiment for inferencing a neural network.

    ``Experiment`` that represents inferencing a neural network using a basic
    inferencing loop.

    This class contains:

    * the data needed for an experiment. The recommended way of setting this
      data is via the arguments of the constructor. Additionally, some of the
      items have getters that are used by the ``Workers`` that execute the
      experiments and by the inferencing loop.
    * the inferencing algorithm, with the main entry point being ``train()``.

    Note:
        When executing a ``BasicInferencing`` in the cloud, additional constraints
        are applied to the data. For example, the model is restricted to
        sequential layers of specific types; the dataset choices are limited,
        etc. Please check the ``CloudRunner`` documentation.
    """

    def __init__(
        self,
        dataset: Type[Dataset],
        model: Module,
        batch_size: int = 10,
        loss_function: type = CrossEntropyLoss,
        weight_template_id: str = "",
        inference_repeats: int = 2,
        inference_time: int = 86400,
        remap_weights: bool = True,
    ):
        """Create a new ``BasicInferencing``.

        Args:
            dataset: the dataset class to be used.
            model: the neural network to use for inferencing.
            batch_size: the batch size used for inferencing.
            loss_function: the loss function used in the neural network.
            weight_template_id: weights and biases of the trained neural network.
            inference_repeats: the number of times running the inference.
            inference_time: the time span between programming the chip and performing the inference.
            remap_weights:  whether to remap the weights
        """
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.inference_repeats = inference_repeats
        self.inference_time = inference_time
        self.weight_template_id = weight_template_id
        self.remap_weights = remap_weights

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
            # mean = Tensor([0.2860])
            # std_dev = Tensor([0.3205])
            # transform = Compose([ToTensor(), Normalize(mean, std_dev)])
            # Note: I removed the normalize step to match up with
            # the steps that were used by Fabio to generate the weight file.
            transform = Compose([ToTensor()])
        elif dataset == SVHN:
            mean = Tensor([0.4377, 0.4438, 0.4728])
            std_dev = Tensor([0.1980, 0.2010, 0.1970])
            transform = Compose([ToTensor(), Normalize(mean, std_dev)])
        else:
            transform = Compose([ToTensor()])

        return transform

    def get_data_loader(
        self,
        dataset: type,
        batch_size: int,
        max_elements: int = 0,
        dataset_root: str = "/tmp/datasets",
    ) -> DataLoader:
        """Return `DataLoaders` for the selected dataset.

        Args:
            dataset: the dataset class to be used.
            batch_size: the batch size used for inferencing.
            max_elements: the maximum number of elements of the dataset
                to be used. If ``0``, the full dataset is used.
            dataset_root: the path to the folder where the files from the
                dataset are stored.

        Returns:
            A tuple with the inferencing and validation loaders.
        """
        # Create the sets and the loaders.
        _, test_args = self.get_dataset_arguments(dataset)
        transform = self.get_dataset_transform(dataset)

        # Create the validation set
        validation_set = dataset(dataset_root, transform=transform, **test_args)

        if max_elements > 0:
            validation_set = Subset(validation_set, range(max_elements))

        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

        return validation_loader

    def get_model(self, weight_template_id: str, device: torch_device) -> Module:
        """Get a copy of the set-up model (with load the weights and biases)
        from the original experiment model.

        Args:
            weight_template_id: location/index for the file
                that contains the state_dicts for the model.
            device: the torch device used for the model.

        Returns:
            a copied model with loaded weights and biases

        """
        model = deepcopy(self.model)

        if weight_template_id != "":
            if weight_template_id[0:1] == "." or weight_template_id[0:1] == "/":
                # This is the case where it is a local file
                template_path = weight_template_id
            else:
                template_dir = "/tmp/weight_templates"
                if weight_template_id.startswith("http"):
                    template_url = weight_template_id
                else:
                    # print('weights_template_id: ', weight_template_id)
                    template_path = template_dir + "/" + weight_template_id + ".pth"
                    template_url = WEIGHT_TEMPLATE_URL + weight_template_id + ".pth"
                # check if the file exists
                if not path.exists(template_dir):
                    mkdir(template_dir)
                if not path.exists(template_path):
                    download(template_url, template_path)

            # print('template_path: ', template_path)
            if path.exists(template_path):
                state_dict = load(template_path, map_location=device, weights_only=False)
                state_dict, _ = convert_legacy_checkpoint(state_dict, model)
                model.load_state_dict(state_dict, load_rpu_config=False)
            else:
                print("Checkpoint file: ", template_path, " does not exist.")

            if self.remap_weights:
                model.remap_analog_weights()

        return model.to(device)

    def inferencing_step(
        self,
        validation_loader: DataLoader,
        model: Module,
        loss_function: _Loss,
        t_inference_list: list,
        device: torch_device,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Run a single inferencing.

        Args:
            validation_loader: the data loader for the inferencing data.
            model: the neural network to be trained.
            loss_function: the loss function used for inferencing.
            t_inference_list: list of t_inferences.
            device: the torch device used for the model.

        Return:
            Tuple of ndarray of inference accuracy, error and loss.
        """

        # Set the mode mode to eval mode.
        model.eval()
        # Reset the program analog weights.
        model.program_analog_weights()

        n_inference = len(t_inference_list)
        infer_accuracy = zeros(n_inference)
        infer_error = zeros(n_inference)
        infer_loss = zeros(n_inference)

        # Simulation of inference pass at different times after training.
        # Go through the generated list
        for idx, t_inference in enumerate(t_inference_list):
            # Set the drift_analog_weights
            model.drift_analog_weights(t_inference)

            # Initialize variables as needed.
            predicted_ok = 0
            total_images = 0
            total_loss = 0

            # Go through the images in the validation dataset
            for images, labels in validation_loader:
                # Load the images and labels into the memory of the device
                images = images.to(device)
                labels = labels.to(device)
                # Do prediction for the images using the model.
                predict = model(images)
                # Calculate the loss
                loss = loss_function(predict, labels)

                # Cummulate the loss to total_loss
                n_images = images.size(0)
                total_images += n_images
                total_loss += loss.item() * n_images

                _, predicted = torch_max(predict.data, 1)
                predicted_ok += (predicted == labels).sum().item()

            # Save the information in the np arrays and return
            accuracy_post = predicted_ok / total_images * 100.0
            infer_accuracy[idx] = accuracy_post
            infer_error[idx] = 100.0 - accuracy_post
            infer_loss[idx] = total_loss / total_images

        return infer_accuracy, infer_error, infer_loss

    def inference(
        self,
        validation_loader: DataLoader,
        model: Module,
        loss_function: _Loss,
        inference_repeats: int,
        inference_time: int,
        device: torch_device,
        n_inference_times: int = 10,
    ) -> Dict:
        """Run the inferencing loop.

        Args:
            validation_loader: the data loader for the validation data.
            model: the neural network to be trained.
            loss_function: the loss function used for inferencing.
            inference_repeats: the number of times to repeat the process
                zof programming and drifting.
            inference_time: the time span between programming the chip and performing the inference.
            device: the torch device used for the model.
            n_inference_times: how many inference times (log-spaced)

        Returns:
            A list of the metrics for each epoch.
        """

        # Move the model to the device if needed.
        if device:
            model = model.to(device)

        # Create the t_inference_list using inference_time.
        # Generate the 9 values between 0 and the inference time using log10
        t_inference_list = [0.0] + logspace(
            0, log10(float(inference_time)), n_inference_times - 1
        ).tolist()
        repeat_results = {}
        accuracy_array = array([], "float")
        error_array = array([], "float")
        loss_array = array([], "float")

        for repeat in range(inference_repeats):
            self._call_hook(Signals.INFERENCE_REPEAT_START, repeat)
            infer_accuracy, infer_error, infer_loss = self.inferencing_step(
                validation_loader, model, loss_function, t_inference_list, device
            )

            # Save the info
            accuracy_array = concatenate([accuracy_array, infer_accuracy])  # type: ignore
            error_array = concatenate([error_array, infer_error])  # type: ignore
            loss_array = concatenate([loss_array, infer_loss])  # type: ignore

            # call the metric hook function with the average information
            # to write out the partial result to standard out.
            shape = (repeat + 1, n_inference_times)
            repeat_results = self._call_hook(
                Signals.INFERENCE_REPEAT_END,
                array(t_inference_list),
                accuracy_array.reshape(shape).mean(axis=0),
                accuracy_array.reshape(shape).std(axis=0),
                error_array.reshape(shape).mean(axis=0),
                loss_array.reshape(shape).mean(axis=0),
                self.inference_repeats,
            )
        return deepcopy(repeat_results)

    def _print_rpu_fields(self, model: Module) -> None:
        """Print the Inference RPU Config fields"""

        print("\n>>> inferenceworker.py: STARTING _print_rpu_fields() ")

        for name, module in model.named_modules():
            if not isinstance(module, AnalogLayerBase):
                continue

            print(f"RPUConfig of module {name}:")
            tile = next(module.analog_tiles())
            print(tile.rpu_config)
            print(tile.tile)
            print("-------------")

        print("\n>>> inferenceworker.py: ENDING _print_rpu_fields() ")

    def run(
        self,
        max_elements: int = 0,
        dataset_root: str = "/tmp/data",
        device: Optional[torch_device] = None,
    ) -> Dict:
        """Sets up the internal model and runs the inference.

        Results are returned and the internal model is updated.
        """

        # Build the objects needed for inferencing.
        # Get valication dataset
        validation_loader = self.get_data_loader(
            self.dataset, self.batch_size, max_elements=max_elements, dataset_root=dataset_root
        )

        # Load the weights and biases to the model.
        # Assumption: the model already includes the customer-specified InferenceRPUConfig.
        model = self.get_model(self.weight_template_id, device)
        self._print_rpu_fields(model)

        # Invoke the inference step
        result = self.inference(
            validation_loader,
            model,
            self.loss_function(),
            self.inference_repeats,
            self.inference_time,
            device,
        )
        self.model = model  # update the stored model with the trained one
        return result

    def __str__(self) -> str:
        """Return a string representation of a BasicInferencing experiment."""
        return (
            "{}(dataset={}, batch_size={}, loss_function={}, inference_repeats={}, "
            "inference_time={}, model={})".format(
                self.__class__.__name__,
                getattr(self.dataset, "__name__", self.dataset),
                self.batch_size,
                getattr(self.loss_function, "__name__", self.loss_function),
                self.inference_repeats,
                self.inference_time,
                self.model,
            )
        )
