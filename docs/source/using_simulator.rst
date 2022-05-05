Using aihwkit Simulator
========================

The core functionality of the package is provided by the ``rpucuda`` simulator.
The simulator contains the primitives and functionality written in C++ and with
CUDA (if enabled), and is exposed to the rest of the package through a Python
interface.

The following table lists the main modules involved in accessing the
simulator:

======================================  ========
Module                                  Notes
======================================  ========
:py:mod:`aihwkit.simulator.tiles`       Entry point for instantiating analog tiles
:py:mod:`aihwkit.simulator.configs`     Configurations and parameters for analog tiles
:py:mod:`aihwkit.simulator.presets`     Presets for analog tiles
:py:mod:`aihwkit.simulator.rpu_base`    Low-level bindings of the C++ simulator members
======================================  ========

Analog Tiles Overview
----------------------

The basic primitives involved in the simulation are **analog tiles**. An
analog tile is a two-dimensional array of **resistive devices** that determine
its behavior and properties, i.e. the material response properties when a single
update pulse is given (a coincidence between row and column pulse train
happened).

The following types of analog tiles are available:

===================================================  ========
Tile class                                           Description
===================================================  ========
:class:`~aihwkit.simulator.tiles.FloatingPointTile`  implements a floating point or ideal analog tile.
:class:`~aihwkit.simulator.tiles.AnalogTile`         implements an abstract analog tile with many cycle-to-cycle non-idealities and systematic parameter-spreads that can be user-defined.
:class:`~aihwkit.simulator.tiles.InferenceTile`      implements an analog tile for inference and hardware-aware training.
===================================================  ========

Creating an Analog Tile
"""""""""""""""""""""""

The simplest way of constructing a tile is by instantiating its class. For
example, the following snippet would create a floating point tile of the
specified dimensions (``10x20``)::

    from aihwkit.simulator.tiles import FloatingPointTile

    tile = FloatingPointTile(10, 20)


GPU-stored Tiles
""""""""""""""""

By default, the ``Tiles`` will be set to perform their computations in the
CPU. They can be moved to the GPU by invoking its ``.cuda()`` method::

    from aihwkit.simulator.tiles import FloatingPointTile

    cpu_tile = FloatingPointTile(10, 20)
    gpu_tile = cpu_tile.cuda()

This method returns a counterpart of its original tile (for example, for a
:class:`~aihwkit.simulator.tiles.FloatingPointTile` it will return a
:class:`~aihwkit.simulator.tiles.CudaFloatingPointTile`). The
GPU-stored tiles share the same interface as the CPU-stored tiled, and their
methods can be used in the same manner.

.. note::

    For GPU-stored tiles to be used, the library needs to be compiled
    with GPU support. This can be checked by inspecting the return value of the
    ``aihwkit.simulator.rpu_base.cuda.is_compiled()`` function.

.. _using-simulator-analog-tiles:

Using Analog Tiles
""""""""""""""""""

Analog arrays are low-level constructs that contain a number of functions that
allow using them in the context of neural networks. A full description of the
available arrays and its methods can be found at
:py:mod:`aihwkit.simulator.tiles`.

Resistive processing units
--------------------------

A **resistive processing unit** is each of the elements on the crossbar array.
The following types of resistive devices are available:

Floating Point Devices
""""""""""""""""""""""

================================================================  ========
Resistive device class                                            Description
================================================================  ========
:class:`~aihwkit.simulator.configs.devices.FloatingPointDevice`   floating point reference, that implements ideal devices forward/backward/update behavior.
================================================================  ========

Single Resistive Devices
""""""""""""""""""""""""

================================================================  ========
Resistive device class                                            Description
================================================================  ========
:class:`~aihwkit.simulator.configs.devices.PulsedDevice`          pulsed update resistive device containing the common properties of all pulsed devices.
:class:`~aihwkit.simulator.configs.devices.IdealDevice`           ideal update behavior (using floating point), but forward/backward might be non-ideal.
:class:`~aihwkit.simulator.configs.devices.ConstantStepDevice`    pulsed update behavioral model: constant step, where the update step of material is constant throughout the resistive range (up to hard bounds).
:class:`~aihwkit.simulator.configs.devices.LinearStepDevice`      pulsed update behavioral model: linear step, where the update step response size of the material is linearly dependent with resistance (up to hard bounds).
:class:`~aihwkit.simulator.configs.devices.SoftBoundsDevice`      pulsed update behavioral model: soft bounds, where the update step response size of the material is linearly dependent and it goes to zero at the bound.
:class:`~aihwkit.simulator.configs.devices.SoftBoundsPmaxDevice`  same model as in :class:`~aihwkit.simulator.configs.devices.SoftBoundsDevice` but using a more convenient parameterization for easier fits to experimentally measured update response curves.
:class:`~aihwkit.simulator.configs.devices.ExpStepDevice`         exponential update step or CMOS-like update behavior.
:class:`~aihwkit.simulator.configs.devices.PowStepDevice`         update step using a power exponent non-linearity.
:class:`~aihwkit.simulator.configs.devices.PiecewiseStepDevice`      user defined device
================================================================  ========

Unit Cell Devices
"""""""""""""""""

====================================================================  ========
Resistive device class                                                Description
====================================================================  ========
:class:`~aihwkit.simulator.configs.devices.VectorUnitCell`            abstract resistive device that combines multiple pulsed resistive devices in a single 'unit cell'.
:class:`~aihwkit.simulator.configs.devices.OneSidedUnitCell`          abstract device model that takes an arbitrary device per crosspoint and implements an explicit plus-minus device pair with one sided update.
:class:`~aihwkit.simulator.configs.devices.ReferenceUnitCell`         abstract device model takes two arbitrary device per cross-point and implements an device with reference pair.
====================================================================  ========

Compound Devices
""""""""""""""""

====================================================================  ========
Resistive device class                                                Description
====================================================================  ========
:class:`~aihwkit.simulator.configs.devices.TransferCompound`          abstract device model that takes 2 or more devices per crosspoint and implements a 'transfer' based learning rule such as Tiki-Taka (see `Gokmen & Haensch 2020`_).
:class:`~aihwkit.simulator.configs.devices.MixedPrecisionCompound`    abstract device model that takes one devices per crosspoint and implements a 'mixed-precision' based learning rule where the rank-update is done in digital instead of using a fully analog parallel write (see `Nandakumar et al. 2020`_).
====================================================================  ========

RPU Configurations
------------------

The combination of the parameters that affect the behavior of a tile and the
parameters that determine the characteristic of a resistive processing unit
are referred to as **RPU configurations**.

Creating a RPU Configuration
""""""""""""""""""""""""""""

A configuration can be created by instantiating the class that corresponds to
the desired tile. Each kind of configuration has different parameters depending
on the particularities of the tile.

For example, for creating a floating point configuration that has the default
values for its parameters::

    from aihwkit.simulator.configs import FloatingPointRPUConfig

    config = FloatingPointRPUConfig()

Among those parameters is the resistive device that will be used for creating
the tile. For example, for creating a single resistive device configuration
that uses a ``ConstantStep`` device::


    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import ConstantStepDevice

    config = SingleRPUConfig(device=ConstantStepDevice())

Device Parameters
"""""""""""""""""

The parameters of the resistive devices that are part of a tile can be set by
passing a ``rpu_config=`` parameter to the constructor::

    from aihwkit.simulator.tiles import AnalogTile
    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import ConstantStepDevice

    config = SingleRPUConfig(device=ConstantStepDevice())
    tile = AnalogTile(10, 20, rpu_config=config)

Each configuration and device have a number of parameters. The parameters can
be specified during the device instantiation, or accessed as attributes of the
device instance.

For example, the following snippet will create a ``LinearStepDevice`` resistive
device, setting its weights limits to ``[-0.4, 0.6]`` and other properties of
the tile::

    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import LinearStepDevice
    from aihwkit.simulator.configs.utils import IOParameters, UpdateParameters

    rpu_config = SingleRPUConfig(
        forward=IOParameters(out_noise=0.1),
        backward=IOParameters(out_noise=0.2),
        update=UpdateParameters(desired_bl=20),
        device=LinearStepDevice(w_min=-0.4, w_max=0.6)
    )

A description of the available parameters each configuration and device can be
found at :py:mod:`aihwkit.simulator.configs`.

An alternative way of specifying non-default parameters is first
generating the config with the correct device and then set the fields directly::

    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import LinearStepDevice

    rpu_config = SingleRPUConfig(device=LinearStepDevice())

    rpu_config.forward.out_noise = 0.1
    rpu_config.backward.out_noise = 0.1
    rpu_config.update.desired_bl = 20
    rpu_config.device.w_min = -0.4
    rpu_config.device.w_max = 0.6

This will generate the same analog tile settings as above.

Unit Cell Device
""""""""""""""""

More complicated devices require specification of sub devices and may
have more parameters. For instance, to configure a device that has 3
resistive device materials per cross-point, which all have different
pulse update behavior, one could do (see also `Example 7`_)::

    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator.configs import UnitCellRPUConfig
    from aihwkit.simulator.configs.utils import VectorUnitCellUpdatePolicy
    from aihwkit.simulator.configs.devices import (
        ConstantStepDevice,
        VectorUnitCell,
        LinearStepDevice,
        SoftBoundsDevice
    )

    # Define a single-layer network, using a vector device having multiple
    # devices per crosspoint. Each device can be arbitrarily defined

    rpu_config = UnitCellRPUConfig()

    rpu_config.device = VectorUnitCell(
        unit_cell_devices=[
            ConstantStepDevice(),
            LinearStepDevice(w_max_dtod=0.4),
            SoftBoundsDevice()
        ]
    )

    # more configurations, if needed

    # only one of the devices should receive a single update that is
    # selected randomly, the effective weights is the sum of all
    # weights
    rpu_config.device.update_policy = VectorUnitCellUpdatePolicy.SINGLE_RANDOM

    # use this configuration for a simple model with one analog tile
    model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)

    # print information about all parameters
    print(model.analog_tile.tile)

This analog tile, although very complicated in its hardware
configuration, can be used in any given network layer in the same way
as simpler analog devices. Also, diffusion or decay, might affect all
sub-devices in difference ways, as they all implement their own
version of these operations. For the vector unit cell, each weight
contribution simple adds up to form a joined effective weight. During
forward/backward this joint effective weight will be used. Update,
however, will be done on each of the "hidden" weights independently.

Transfer Compound Device
""""""""""""""""""""""""
Compound devices are more complex than unit cell devices, which have a
number of devices per crosspoint, however, they share the underlying
implementation. For instance, the "Transfer Compound Device" does
contain (at least) two full crossbar arrays internally, where the
stochastic gradient descent update is done on one (or a subset of
these). It does a partial transfer of content in the first array to the
second intermittently. This transfer is accomplished by doing an
extra forward pass (with a one-hot input vector) on the first array
and updating the output onto the second array. The parameter of this
extra forward and update step can be given.

This compound device can be used to implement the tiki-taka learning
rule as described in `Gokmen & Haensch 2020`_. For instance, one could
use the following tile configuration for that (see also `Example 8`_)::


    # Imports from aihwkit.
    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator.configs import UnitCellRPUConfig
    from aihwkit.simulator.configs.devices import (
        TransferCompound,
        SoftBoundsDevice
    )

    # The Tiki-taka learning rule can be implemented using the transfer device.
    rpu_config = UnitCellRPUConfig(
        device=TransferCompound(

            # devices that compose the Tiki-taka compound
            unit_cell_devices=[
                SoftBoundsDevice(w_min=-0.3, w_max=0.3),
                SoftBoundsDevice(w_min=-0.6, w_max=0.6)
            ],

            # Make some adjustments of the way Tiki-Taka is performed.
            units_in_mbatch=True,   # batch_size=1 anyway
            transfer_every=2,       # every 2 batches do a transfer-read
            n_cols_per_transfer=1,  # one forward read for each transfer
            gamma=0.0,              # all SGD weight in second device
            scale_transfer_lr=True, # in relative terms to SGD LR
            transfer_lr=1.0,        # same transfer LR as for SGD
        )
    )

    # make more adjustments (can be made here or above)
    rpu_config.forward.inp_res = 1/64. # 6 bit DAC

    # same forward/update for transfer-read as for actual SGD
    rpu_config.device.transfer_forward = rpu_config.forward

    # SGD update/transfer-update will be done with stochastic pulsing
    rpu_config.device.transfer_update = rpu_config.update

    # use tile configuration in model
    model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)

    # print some parameter infos
    print(model.analog_tile.tile)


Note that this analog tile now will perform tiki-taka as the learning
rule instead of plain SGD. Once the configuration is done, the usage
of this complex analog tile for testing or training from the user
point of view is however the same as for other tiles.

Mixed Precision Compound
""""""""""""""""""""""""

This abstract device implements an analog SGD optimizer suggested by
`Nandakumar et al. 2020`_ where the update is not done in analog
directly, but in digital. Thus is uses a digital rank-update of an
intermediately stored floating point matrix, which will be used to
transfer the information to the analog tile that is used in forward
and backward pass.  This optimizer strategy is in contrast with the
default mode in the simulator, that uses stochastic pulse trains to
update in parallel onto the analog tile directly. This will have
impact on the hardware design as well as expected runtime, as more
digital computation is needed to be done. For details, see `Nandakumar
et al. 2020`_.

To enable mixed-precision one defines for example the following ``rpu_config``::

    # Imports from aihwkit.
    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator.configs import DigitalRankUpdateRPUConfig
    from aihwkit.simulator.configs.devices import (
        SoftBoundsDevice, MixedPrecisionCompound
    )

    rpu_config = DigitalRankUpdateRPUConfig(
        device=MixedPrecisionCompound(
            device=SoftBoundsDevice(),

            # make some adjustments of mixed-precision hyper parameter
            granularity=0.001,
            n_x_bins=0,  # floating point actiations for Chi update
            n_d_bins=0,  # floating point delta for Chi update
	)
    )

    # use tile configuration in model
    model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)

Now this analog tile will use the mixed-precision optimizer with a
soft bounds device model.

Analog Presets
--------------

In addition to the building blocks for analog tiles described in the sections
above, the toolkit includes:

* a library of device presets that are calibrated to real hardware data and/or
  are based on models in the literature.
* a library of configuration presets that specify a particular device and
  optimizer choice.

The current list of device and configuration presets can be found in the
:py:mod:`aihwkit.simulator.presets` module. These presets can be used directly
instead of manually specifying a ``RPU Configuration``::

    from aihwkit.simulator.tiles import AnalogTile
    from aihwkit.simulator.presets import TikiTakaEcRamPreset

    tile = AnalogTile(10, 20, rpu_config=TikiTakaEcRamPreset())


.. _Gokmen & Haensch 2020: https://www.frontiersin.org/articles/10.3389/fnins.2020.00103/full
.. _Example 7: https://github.com/IBM/aihwkit/blob/master/examples/07_simple_layer_with_other_devices.py
.. _Example 8: https://github.com/IBM/aihwkit/blob/master/examples/08_simple_layer_with_tiki_taka.py
.. _Nandakumar et al. 2020: https://www.frontiersin.org/articles/10.3389/fnins.2020.00406/full
