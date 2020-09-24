Using analog tiles
==================

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
:py:mod:`aihwkit.simulator.devices`     Entry point for instantiating resistive devices
:py:mod:`aihwkit.simulator.parameters`  Different parameters used by the resistive devices
:py:mod:`aihwkit.simulator.rpu_base`    Low-level bindings of the C++ simulator members
======================================  ========

Analog tiles and resistive devices
----------------------------------

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
===================================================  ========

And the following types of resistive devices are available:

================================================================  ========
Resistive device class                                            Description
================================================================  ========
:class:`~aihwkit.simulator.devices.FloatingPointResistiveDevice`  floating point reference, that implements ideal devices forward/backward/update behavior.
:class:`~aihwkit.simulator.devices.PulsedResistiveDevice`         pulsed update resistive device containing the common properties of all pulsed devices.
:class:`~aihwkit.simulator.devices.IdealResistiveDevice`          ideal update behavior (using floating point), but forward/backward might be non-ideal.
:class:`~aihwkit.simulator.devices.ConstantStepResistiveDevice`   pulsed update behavioral model: constant step, where the update step of material is constant throughout the resistive range (up to hard bounds).
:class:`~aihwkit.simulator.devices.LinearStepResistiveDevice`     pulsed update behavioral model: linear step, where the update step response size of the material is linearly dependent with resistance (up to hard bounds).
:class:`~aihwkit.simulator.devices.SoftBoundsResistiveDevice`     pulsed update behavioral model: soft bounds, where the update step response size of the material is linearly dependent and it goes to zero at the bound.
:class:`~aihwkit.simulator.devices.ExpStepResistiveDevice`        exponential update step or CMOS-like update behavior.
:class:`~aihwkit.simulator.devices.VectorUnitCell`                abstract resistive device that combines multiple pulsed resistive devices in a single 'unit cell'.
:class:`~aihwkit.simulator.devices.DifferenceUnitCell`            abstract device model takes an arbitrary device per crosspoint and implements an explicit plus-minus device pair.
:class:`~aihwkit.simulator.devices.TransferUnitCell`              abstract device model that takes 2 or more devices per crosspoint and implements a 'transfer' based learning rule such as Tiki-Taka (see `Gokmen & Haensch 2020`_).
================================================================  ========


Creating an analog tile
-----------------------

The simplest way of constructing a tile is by instantiating its class. For
example, the following snippet would create a floating point tile of the
specified dimensions (``10x20``)::

    from aihwkit.simulator.tiles import FloatingPointTile

    tile = FloatingPointTile(10, 20)

The parameters of the resistive devices that are part of a tile can be set by
passing a ``resistive_device=`` parameter to the constructor::

    from aihwkit.simulator.tiles import AnalogTile
    from aihwkit.simulator.devices import ConstantStepResistiveDevice

    device = ConstantStepResistiveDevice()
    tile = AnalogTile(10, 20, device)

Analog arrays are low-level constructs that contain a number of functions that
allow using them in the context of neural networks. A full description of the
available arrays and its methods can be found at
:py:mod:`aihwkit.simulator.tiles`.

GPU-stored tiles
~~~~~~~~~~~~~~~~~

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

Specifying resistive devices
----------------------------

Each resistive device has a number of parameters an options that determines
its behavior. A resistive device can be created by instantiating the
corresponding class.

For example, for creating a floating point device that has the default values
for its parameters::

    from aihwkit.simulator.devices import FloatingPointResistiveDevice

    device = FloatingPointResistiveDevice()


Device parameters
~~~~~~~~~~~~~~~~~

The behavior of a device is controlled by its parameters. The parameters can
be specified during the device instantiation, or accessed as attributes of the
device instance.

For example, the following snipped will create a ``ConstantStep`` resistive
device, setting its weigths limits to ``[-0.4, 0.6]``::

    from aihwkit.simulator.devices import ConstantStepResistiveDevice
    from aihwkit.simulator.parameters import PulsedResistiveDeviceParameters

    parameters = PulsedResistiveDeviceParameters(w_min=-0.4)
    device = ConstantStepResistiveDevice(parameters)
    device.params.w_max = 0.6

A description of the available parameters for each device can be found at
:py:mod:`aihwkit.simulator.parameters`.

.. _Gokmen & Haensch 2020: https://www.frontiersin.org/articles/10.3389/fnins.2020.00103/full
