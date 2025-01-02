# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for optimizers."""

from torch import Tensor, manual_seed
from torch.nn import Linear
from torch.optim import SGD, AdamW
from torch.nn.functional import mse_loss

from aihwkit.optim import AnalogOptimizer

from .helpers.testcases import AihwkitTestCase


class InferenceOptimizerTest(AihwkitTestCase):
    """Tests for the AnalogOptimizer."""

    @staticmethod
    def train_model(model, opt, loss_func, x_b, y_b):
        """Train the model."""
        epochs = 100
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x_b)
            loss = loss_func(pred, y_b)

            loss.backward()
            opt.step()

        return loss

    def test_instantiate(self):
        """Test instantiating the optimizer."""
        model = Linear(3, 4)
        optimizer = AnalogOptimizer(SGD, model.parameters(), lr=0.123)

        # Assert that a new subclass is created, with both the wrapped
        # optimizer and the AnalogInferenceOptimizer as parents.
        self.assertIsInstance(optimizer, AnalogOptimizer)
        self.assertIsInstance(optimizer, SGD)
        self.assertIsNot(type(optimizer), AnalogOptimizer)

        # Assert over specific wrapped optimizer parameters.
        self.assertEqual(optimizer.defaults["lr"], 0.123)

    def test_train_digital_sgd(self):
        """Test training digital layer with analog optimizer (SGD)."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.1], [0.2, 0.4, 0.3, 0.6]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model1 = Linear(4, 1)

        manual_seed(4321)
        model2 = Linear(4, 1)

        learning_rate = 0.123
        optimizer1 = AnalogOptimizer(SGD, model1.parameters(), lr=learning_rate)
        optimizer2 = SGD(model2.parameters(), lr=learning_rate)

        loss1 = self.train_model(model1, optimizer1, loss_func, x_b, y_b)
        loss2 = self.train_model(model2, optimizer2, loss_func, x_b, y_b)

        self.assertAlmostEqual(loss1, loss2)
        self.assertTensorAlmostEqual(model1.weight, model2.weight)

    def test_train_digital_adamw(self):
        """Test training digital layer with analog optimizer (AdamW)."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.1], [0.2, 0.4, 0.3, 0.6]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model1 = Linear(4, 1)

        manual_seed(4321)
        model2 = Linear(4, 1)

        learning_rate = 0.123
        optimizer1 = AnalogOptimizer(AdamW, model1.parameters(), lr=learning_rate)
        optimizer2 = AdamW(model2.parameters(), lr=learning_rate)

        loss1 = self.train_model(model1, optimizer1, loss_func, x_b, y_b)
        loss2 = self.train_model(model2, optimizer2, loss_func, x_b, y_b)

        self.assertAlmostEqual(loss1, loss2)
        self.assertTensorAlmostEqual(model1.weight, model2.weight)
