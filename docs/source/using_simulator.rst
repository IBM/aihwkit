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
:py:mod:`aihwkit.simulator.configs`     Configurations and parameters for analog tiles
:py:mod:`aihwkit.simulator.rpu_base`    Low-level bindings of the C++ simulator members
======================================  ========

Analog tiles
------------

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

Creating an analog tile
"""""""""""""""""""""""

The simplest way of constructing a tile is by instantiating its class. For
example, the following snippet would create a floating point tile of the
specified dimensions (``10x20``)::

    from aihwkit.simulator.tiles import FloatingPointTile

    tile = FloatingPointTile(10, 20)


GPU-stored tiles
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

Using analog tiles
""""""""""""""""""

Analog arrays are low-level constructs that contain a number of functions that
allow using them in the context of neural networks. A full description of the
available arrays and its methods can be found at
:py:mod:`aihwkit.simulator.tiles`.

Resistive processing units
--------------------------

A **resistive processing unit** is each of the elements on the crossbar array.
The following types of resistive devices are available:

Floating point devices
""""""""""""""""""""""

================================================================  ========
Resistive device class                                            Description
================================================================  ========
:class:`~aihwkit.simulator.configs.devices.FloatingPointDevice`   floating point reference, that implements ideal devices forward/backward/update behavior.
================================================================  ========

Single resistive devices
""""""""""""""""""""""""

================================================================  ========
Resistive device class                                            Description
================================================================  ========
:class:`~aihwkit.simulator.configs.devices.PulsedDevice`          pulsed update resistive device containing the common properties of all pulsed devices.
:class:`~aihwkit.simulator.configs.devices.IdealDevice`           ideal update behavior (using floating point), but forward/backward might be non-ideal.
:class:`~aihwkit.simulator.configs.devices.ConstantStepDevice`    pulsed update behavioral model: constant step, where the update step of material is constant throughout the resistive range (up to hard bounds).
:class:`~aihwkit.simulator.configs.devices.LinearStepDevice`      pulsed update behavioral model: linear step, where the update step response size of the material is linearly dependent with resistance (up to hard bounds).
:class:`~aihwkit.simulator.configs.devices.SoftBoundsDevice`      pulsed update behavioral model: soft bounds, where the update step response size of the material is linearly dependent and it goes to zero at the bound.
:class:`~aihwkit.simulator.configs.devices.ExpStepDevice`         exponential update step or CMOS-like update behavior.
================================================================  ========

Unit cell devices
"""""""""""""""""

====================================================================  ========
Resistive device class                                                Description
====================================================================  ========
:class:`~aihwkit.simulator.configs.devices.VectorUnitCellDevice`      abstract resistive device that combines multiple pulsed resistive devices in a single 'unit cell'.
:class:`~aihwkit.simulator.configs.devices.DifferenceUnitCellDevice`  abstract device model takes an arbitrary device per crosspoint and implements an explicit plus-minus device pair.
====================================================================  ========

Compound devices
""""""""""""""""

====================================================================  ========
Resistive device class                                                Description
====================================================================  ========
:class:`~aihwkit.simulator.configs.devices.TransferUnitCellDevice`    abstract device model that takes 2 or more devices per crosspoint and implements a 'transfer' based learning rule such as Tiki-Taka (see `Gokmen & Haensch 2020`_).
====================================================================  ========

RPU Configurations
------------------

The combination of the parameters that affect the behavior of a tile and the
parameters that determine the characteristic of a resistive processing unit
are referred to as **RPU configurations**.

Creating a RPU configuration
""""""""""""""""""""""""""""

A configuration can be created by instantiating the class that corresponds to
the desired tile. Each kind of configuration has different parameters depending
on the particularities of the tile.

For example, for creating a floating point configuration that has the default
values for its parameters::

    from aihwkit.simulator.configs import FloatingPointResistiveDevice

    config = FloatingPointResistiveDevice()

Among those parameters is the resistive device that will be used for creating
the tile. For example, for creating a single resistive device configuration
that uses a ``ConstantStep`` device::


    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import ConstantStepDevice

    config = SingleRPUConfig(device=ConstantStepDevice())

Device parameters
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

    rpu_config = SingleRPUConfig(
        forward=IOParameters(inp_noise=0.1),
        backward=BackwardIOParameters(inp_noise=0.2),
        update=UpdateParameters(desired_bl=60),
        device=LinearStepDevice(w_min=-0.4, w_max=0.6)
    )

A description of the available parameters each configuration and device can be
found at :py:mod:`aihwkit.simulator.configs`.

.. _Gokmen & Haensch 2020: https://www.frontiersin.org/articles/10.3389/fnins.2020.00103/full
