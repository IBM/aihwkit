# IBM Analog Hardware Acceleration Kit: Examples

We have many different examples to explore the many features of the IBM Analog Hardware
Acceleration Kit:

## Example 1: [`1_simple_layer.py`]

In this example a single fully connected analog layer is used to predict the output tensor y, based
on the input tensor x. The `rpu_config parameter` of the `AnalogLinear` layer can be used  to
define different settings for the network. In this case it specifies the device
`ConstantStepDevice`, but different [Resistive Processing Units] exist:

```python
# Define a single-layer network, using a constant step device type.
rpu_config = SingleRPUConfig(device=ConstantStepDevice())
model = AnalogLinear(4, 2, bias=True,
                     rpu_config=rpu_config)

```

In the first few examples both training and inference are done on an analog chip, while few of the
later examples will use the Hardware Aware (HWA) training. HWA training uses digital chip for the
training part and analog one for the inference one. The network is trained for 100 epochs using
the analog Stochastic Gradient Descent and the loss is printed for every epoch:

```
Loss error: 0.5686419010162354
...
Loss error: 0.0137629760429263

Process finished with exit code 0
```

## Example 2: [`2_multiple_layer.py`]

The second example uses a larger fully connected network to predict the output tensor y based on
the input tensor x. In this network the multiple fully connected analog layer are wrapped by a
sequential container. Similarly to the first example, the network uses a `ConstantStepDevice` and
it is  trained for 100 epochs with the loss printed at every epoch.

## Example 3: [`3_minst_training.py`]

This MNIST training example is based on the paper: [Gokmen T and Vlasov Y (2016) Acceleration of
Deep Neural Network Training with Resistive Cross-Point Devices: Design Considerations. Front.
Neurosci.].
It builds a deep neural network of fully connected layers with 784, 256, 128 and 10 neurons,
sigmoids and softmax activation functions.

```
Sequential(
  (0): AnalogLinear(in_features=784, out_features=256, bias=True, is_cuda=False)
  (1): Sigmoid()
  (2): AnalogLinear(in_features=256, out_features=128, bias=True, is_cuda=False)
  (3): Sigmoid()
  (4): AnalogLinear(in_features=128, out_features=10, bias=True, is_cuda=False)
  (5): LogSoftmax()
)
```

The device used in the analog tile is a `ConstantStepDevice`. The network is trained with a
standard MNIST dataset with a Stochastic Gradient Descent optimizer. The network is trained for 30
epochs with batch size of 64 and a fixed learning rate of 0.05 and the loss is printed at every
epoch.

## Example 4: [`4_lenet5_training.py`]

This CNN MNIST training example is based on the paper: [Gokmen T, Onen M and Haensch W (2017)
Training Deep Convolutional Neural Networks with Resistive Cross-Point Devices. Front. Neurosci.].
It is a deep neural network of convolutional layers with an architecture similar to LeNet5.
It has 2 convolutional layers with 5x5 kernels, tanh and softmax activation functions and uses
max-pooling as subsampling layer. The full architecture is printed in terminal during execution:

```
LeNet5(
  (feature_extractor): Sequential(
    (0): AnalogConv2d(1, 16, kernel_size=(5, 5), stride=(1, 1))
    (1): Tanh()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): AnalogConv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
    (4): Tanh()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Tanh()
  )
  (classifier): Sequential(
    (0): AnalogLinear(in_features=512, out_features=128, bias=True, is_cuda=True)
    (1): Tanh()
    (2): AnalogLinear(in_features=128, out_features=10, bias=True, is_cuda=True)
  )
)
```

This example can be used to evaluate the accuracy difference between training with an analog
network (specified by `ConstantStepDevice`) or a digital network (specified by
`FloatingPointDevice`), by selecting the proper `SingleRPUConfig` definition:

```python
# Select the device model to use in the training.
# * If `SingleRPUConfig(device=ConstantStepDevice())` then analog tiles with
#   constant step devices will be used,
# * If `FloatingPointRPUConfig(device=FloatingPointDevice())` then standard
#   floating point devices will be used
RPU_CONFIG = SingleRPUConfig(device=ConstantStepDevice())
# RPU_CONFIG = FloatingPointRPUConfig(device=FloatingPointDevice())
```

The CNN is trained for 30 epochs with a batch size of 8, a Stochastic Gradient Descent optimizer
and a learning rate of 0.01. When training on GPU (V100) it takes ~20s/epoch to train the network;
when working with CPU, the time required for training with a `ConstantSetpDevice` is substantially
longer (on a standard laptop ~5min/epoch) than `FloatingPointDevice` is used. The timestamp,
losses, test error and accuracy are printed for every epoch:

```
10:20:13 --- Started LeNet5 Example
10:20:32 --- Epoch: 0	Train loss: 1.9803	Valid loss: 1.6184	Test error: 10.66%	Accuracy: 89.34%
...
10:29:27 --- Epoch: 29	Train loss: 1.4724	Valid loss: 1.4768	Test error: 1.32%	Accuracy: 98.68%
10:29:27 --- Completed LeNet5 Example
```

At the end of the training/evaluation phases of the network, the losses, accuracy and test_error
are also saved to file in a `results/LENET5/` folder (from the path where the example was
executed) together with a plot that shows the training evolution over the multiple epochs.

![Test Losses](img/test_losses.png)

![Test Error](img/test_error.png)

## Example 5: [`5_simple_layer_hardware_aware.py`]

Templating from the example 1, this example use the same input/output tensors and same architecture,
but uses the hardware aware training functionality. In hardware aware training the network training
is performed on a digital network, with many noise sources typical of Phase Change Memory (PCM)
inserted in this phase to make the network more resilient to these. The inference part is instead
performed on the analog network to take full advantage of the speed of the analog computation.
Many features and parameters can be defined in the `rpu_config` to specify the noise sources
characteristics:

```python
# Define a single-layer network, using inference/hardware-aware training tile
rpu_config = InferenceRPUConfig()
rpu_config.forward.out_res = -1.  # Turn off (output) ADC discretization.
rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
rpu_config.forward.w_noise = 0.02  # Short-term w-noise.

rpu_config.clip.type = WeightClipType.FIXED_VALUE
rpu_config.clip.fixed_value = 1.0
rpu_config.modifier.pdrop = 0.03  # Drop connect.
rpu_config.modifier.type = WeightModifierType.ADD_NORMAL  # Fwd/bwd weight noise.
rpu_config.modifier.std_dev = 0.1
rpu_config.modifier.rel_to_actual_wmax = True

# Inference noise model.
rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
```
Additionally this example explore the use of `drift_compensation` methods to improve the accuracy
of the analog network during the inference phase in presence of [weight drift]. The
`drift_compensation` is also specified in the `rpu_config` option:

```python
# drift compensation
rpu_config.drift_compensation = GlobalDriftCompensation()
```
The weight drift is exercised through the drift_analog_weights function that is applied to the
analog model, and it can be looped over to explore inference accuracy at different times since
training:

```python
for t_inference in [0., 1., 20., 1000., 1e5]:
    model.drift_analog_weights(t_inference)
    pred_drift = model(x)
    print('Prediction after drift (t={}, correction={:1.3f}):\t {}'.format(
        t_inference, model.analog_tile.alpha.cpu().numpy(),
        pred_drift.detach().cpu().numpy().flatten()))
```

More details are discussed in the [Inference and PCM statistical model] documentation.

## Example 6: [`6_lenet5_hardware_aware.py`]

This example templates from the CNN of example 4 and add noise and non-idealities typical of the
PCM.

```python
# Define the properties of the neural network in terms of noise simulated during
# the inference/training pass
RPU_CONFIG = InferenceRPUConfig()
RPU_CONFIG.forward.out_res = -1.  # Turn off (output) ADC discretization.
RPU_CONFIG.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
RPU_CONFIG.forward.w_noise = 0.02
RPU_CONFIG.noise_model = PCMLikeNoiseModel(g_max=25.0)
```

As the previous example it then explores the effect of the PCM weight drift on the inference
accuracy:

```python
    # Simulation of inference pass at different times after training.
    for t_inference in [0., 1., 20., 1000., 1e5]:
        model.drift_analog_weights(t_inference)
```

## Example 7: [`7_simple_layer_with_other_devices.py`]

This example templates from example 1. However, rather than having a single device at every
cross-point of the RPU array it defines 3 different devices, each one with its specific parameters,
and the total weights is represented by the sum of the weights of these 3 devices:

```python
# 3 arbitrary single unit cell devices (of the same type) per cross-point.
rpu_config.device = VectorUnitCell(
    unit_cell_devices=[
        ConstantStepDevice(w_max=0.3),
        ConstantStepDevice(w_max_dtod=0.4),
        ConstantStepDevice(up_down_dtod=0.1),
    ])
```

The example also define the specific update policy that is used to learn the weights of the
network during the training phase. In this example the update is ony done on a single device which
is randomly selected among the 3:

```python
# Only one of the devices should receive a single update.
# That is selected randomly, the effective weights is the sum of all
# weights.
rpu_config.device.update_policy = VectorUnitCellUpdatePolicy.SINGLE_RANDOM
```

More information can be find in the [Unit Cell Device] documentation. Similarly to example 1 the
network is trained over 100 epochs with an analog Stochastic Gradient Descent optimizer and the
loss is printed for every epoch.

## Example 8: [`8_simple_layer_with_tiki_taka.py`]

This example aims at exploring a training algorithm specifically developed for RPU and resistive
device and discussed in [Gokmen T and Haensch W (2020) Algorithm for Training Neural Networks on
Resistive Device Arrays. Front. Neurosci.].
The Tiki-Taka algorithm relaxes the symmetricity requirements on the switching characteristic of
resistive devices used in RPU cross-point array. In this example 2 [Compound Device] are defined
at each crosspoint and the Tiki-Taka algorithm parameters are specified as part of the
```UnitCellRPUConfig```:

```python
# The Tiki-taka learning rule can be implemented using the transfer device.
rpu_config = UnitCellRPUConfig(
    device=TransferCompound(

        # Devices that compose the Tiki-taka compound.
        unit_cell_devices=[
            SoftBoundsDevice(w_min=-0.3, w_max=0.3),
            SoftBoundsDevice(w_min=-0.6, w_max=0.6)
        ],

        # Make some adjustments of the way Tiki-Taka is performed.
        units_in_mbatch=True,    # batch_size=1 anyway
        transfer_every=2,        # every 2 batches do a transfer-read
        n_cols_per_transfer=1,   # one forward read for each transfer
        gamma=0.0,               # all SGD weight in second device
        scale_transfer_lr=True,  # in relative terms to SGD LR
        transfer_lr=1.0,         # same transfer LR as for SGD
    )
)
```

Similarly to example 1 the network is trained over 100 epochs with an analog Stochastic Gradient
Descent optimizer and the loss is printed for every epoch.

[Apache License 2.0]: LICENSE.txt
[Resistive Processing Units]: https://aihwkit.readthedocs.io/en/latest/using_simulator.html#resistive-processing-units
[Inference and PCM statistical model]: https://aihwkit.readthedocs.io/en/latest/pcm_inference.html
[Unit Cell Device]: https://aihwkit.readthedocs.io/en/latest/using_simulator.html#unit-cell-device
[Compound Device]: https://aihwkit.readthedocs.io/en/latest/using_simulator.html#transfer-compound-device
[weight drift]: https://aihwkit.readthedocs.io/en/latest/pcm_inference.html#drift

[Gokmen T and Vlasov Y (2016) Acceleration of Deep Neural Network Training with Resistive
Cross-Point Devices: Design Considerations. Front. Neurosci.]:
https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full
[Gokmen T, Onen M and Haensch W (2017) Training Deep Convolutional Neural Networks with
Resistive Cross-Point Devices. Front. Neurosci.]:
https://www.frontiersin.org/articles/10.3389/fnins.2017.00538/full
[Gokmen T and Haensch W (2020) Algorithm for Training Neural Networks on Resistive Device Arrays.
Front. Neurosci.]: https://www.frontiersin.org/articles/10.3389/fnins.2020.00103/full

[`1_simple_layer.py`]: 1_simple_layer.py
[`2_multiple_layer.py`]: 2_multiple_layer.py
[`3_minst_training.py`]: 3_minst_training.py
[`4_lenet5_training.py`]: 4_lenet5_training.py
[`5_simple_layer_hardware_aware.py`]: 5_simple_layer_hardware_aware.py
[`6_lenet5_hardware_aware.py`]: 6_lenet5_hardware_aware.py
[`7_simple_layer_with_other_devices.py`]: 7_simple_layer_with_other_devices.py
[`8_simple_layer_with_tiki_taka.py`]: 8_simple_layer_with_tiki_taka.py
