# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog], and this project adheres to
[Semantic Versioning]:

* `Added` for new features.
* `Changed` for changes in existing functionality.
* `Deprecated` for soon-to-be removed features.
* `Removed` for now removed features.
* `Fixed` for any bug fixes.
* `Security` in case of vulnerabilities.

## [UNRELEASED]
* A number of new config presets added to the library, namely `EcRamMOPreset`, `EcRamMO2Preset`,
  `EcRamMO4Preset`, `TikiTakaEcRamMOPreset`, `MixedPrecisionEcRamMOPreset`. These can be used for 
  tile configuration (`rpu_config`). They specify a particular device and optimizer choice.

## [0.3.0] - 2021/04/14

### Added

* New analog devices:
  * A new abstract device (`MixedPrecisionCompound`) implementing an SGD
    optimizer that computes the rank update in digital (assuming digital
    high precision storage) and then transfers the matrix sequentially to
    the analog device, instead of using the default fully parallel pulsed
    update. (\#159)
  * A new device model class `PowStepDevice` that implements a power-exponent
    type of non-linearity based on the Fusi & Abott synapse model. (\#192)
  * New parameterization of the `SoftBoundsDevice`, called
    `SoftBoundsPmaxDevice`. (\#191)
* Analog devices and tiles improvements:
  * Option to choose deterministic pulse trains for the rank-1 update of
    analog devices during training. (\#99)
  * More noise types for hardware-aware training for inference
    (polynomial). (\#99)
  * Additional bound management schemes (worst case, average max, shift).
    (\#99)
  * Cycle-to-cycle output referred analog multiply-and-accumulate weight
    noise that resembles the conductance dependent PCM read noise
    statistics. (\#99)
  * C++ backend improvements (slice backward/forward/update, direct
    update). (\#99)
  * Option to excluded bias row for hardware-aware training noise. (\#99)
  * Option to automatically scale the digital weights into the full range of
    the simulated crossbar by applying a fixed output global factor in
    digital. (\#129)
  * Optional power-law drift during analog training. (\#158)
  * Cleaner setting of `dw_min` using device granularity. (\#200)
* PyTorch interface improvements:
  * Two new convolution layers have been added: `AnalogConv1d` and
    `AnalogConv3d`, mimicking their digital counterparts. (\#102, \#103)
  * The `.to()` method can now be used in `AnalogSequential`, along with
    `.cpu()` methods in analog layers (albeit GPU to CPU is still not
    possible). (\#142, \#149)
* New modules added:
  * A library of device presets that are calibrated to real hardware data,
    namely `ReRamESPresetDevice`, `ReRamSBPresetDevice`, `ECRamPresetDevice`,
    `CapacitorPresetDevice`, and device presets that are based on models in the
    literature, e.g. `GokmenVlasovPresetDevice` and `IdealizedPresetDevice`.
    They can be used defining the device field in the `RPUConfig`. (\#144)
  * A library of config presets, such as `ReRamESPreset`, `Capacitor2Preset`,
    `TikiTakaReRamESPreset`, and many more. These can be used for tile
    configuration (`rpu_config`). They specify a particular device and optimizer
    choice. (\#144)
  * Utilities for visualization the pulse response properties of a given
    device configuration. (\#146)
  * A new `aihwkit.experiments` module has been added that allows creating and
    running specific high-level use cases (for example, neural network training)
    conveniently. (\#171, \#172)
  * A `CloudRunner` class has been added that allows executing experiments in
    the cloud. (\#184)

#### Changed

* The minimal PyTorch version has been bumped to `1.7+`. Please recompile your
  library and update the dependencies accordingly. (\#176)
* Default value for TransferCompound for `transfer_every=0` (\#174).

#### Fixed

* Issue of number of loop estimations for realistic reads. (\#192)
* Fixed small issues that resulted in warnings for windows compilation. (\#99)
* Faulty backward noise management error message removed for perfect backward
  and CUDA. (\#99)
* Fixed segfault when using diffusion or reset with vector unit cells for
  CUDA. (\#129)
* Fixed random states mismatch in IoManager that could cause crashed in same
  network size and batch size cases for CUDA, in particular for
  `TransferCompound`. (\#132)
* Fixed wrong update for `TransferCompound` in case of `transfer_every` smaller
  than the batch size. (\#132, \#174)
* Period in the modulus of `TransferCompound` could become zero which
  caused a floating point exception. (\#174)
* Ceil instead of round for very small transfers in `TransferCompound`
  (to avoid zero transfer for extreme settings). (\#174)

#### Removed

* The legacy `NumpyAnalogTile` and `NumpyFloatingPointTile` tiles have been
  finally removed. The regular, tensor-powered `aihwkit.simulator.tiles` tiles
  contain all their functionality and numerous additions. (\#122)

## [0.2.1] - 2020/11/26

* The `rpu_config` is now pretty-printed in a readable manner (excluding the
  default settings and other readability tweak). (\#60)
* Added a new `ReferenceUnitCell` which has two devices, where one is fixed and
  the other updated and the effective weight is computed a difference between
  the two. (\#61)
* `VectorUnitCell` accepts now arbitrary weighting schemes that can be
  user-defined by using a new `gamma_vec` property that specifies how to combine
  the unit cell devices to form the effective weight. (\#61)

### Changed

* The unit cell items in `aihwkit.simulator.configs` have been renamed, removing
  their `Device` suffix, for having a more consistent naming scheme. (\#57)
* The `Exceptions` raised by the library have been revised, making use in some
  cases of the ones introduced in a new `aihwkit.exceptions` module. (\#49)
* Some `VectorUnitCell` properties have been renamed and extended with an update
  policy specifying how to select the hidden devices. (\#61)
* The `pybind11` version required has been bumped to 2.6.0, which can be
  installed from `pip` and makes system-wide installation no longer required.
  Please update your `pybind11` accordingly for compiling the library. (\#44)

### Removed

* The `BackwardIOParameters` specialization has been removed, as bound
  management is now automatically ignored for the backward pass. Please use the
  more general `IOParameters` instead. (\#45)

### Fixed

* Serialization of `Modules` that contain children analog layers is now
  possible, both when using containers such as `Sequential` and when using
  analog layers as custom Module attributes. (\#74, \#80)
* The build system has been improved, with experimental Windows support and
  supporting using CUDA 11 correctly. (\#58, \#67, \#68)

## [0.2.0] - 2020/10/20

### Added

* Added more types of resistive devices: `IdealResistiveDevice`, `LinearStep`,
  `SoftBounds`, `ExpStep`, `VectorUnitCell`, `TransferCompoundDevice`,
  `DifferenceUnitCell`. (\#14)
* Added a new `InferenceTile` that supports basic hardware-aware training
  and inference using a statistical noise model that was fitted by real PCM
  devices. (\#25)
* Added a new `AnalogSequential` layer that can be used in place of `Sequential`
  for easier operation on children analog layers. (\#34)

### Changed

* Specifying the tile configuration (resistive device and the rest of the
  properties) is now based on a new `RPUConfig` family of classes, that is
  passed as a `rpu_config` argument instead of `resistive_device` to `Tiles`
  and `Layers`. Please check the `aihwkit.simulator.config` module for more
  details. (\#23)
* The different analog tiles are now organized into a `aihwkit.simulator.tiles`
  package. The internal `IndexedTiles` have been removed, and the rest of
  previous top-level imports have been kept. (\#29)

### Fixed

* Improved package compatibility when using non-UTF8 encodings (version file,
  package description). (\#13)
* The build system can now detect and use `openblas` directly when using the
  conda-installable version. (\#22)
* When using analog layers as children of another module, the tiles are now
  correctly moved to CUDA if using `AnalogSequential` (or by the optimizer if
  using regular torch container modules). (\#34)

## [0.1.0] - 2020/09/17

### Added

* Initial public release.
* Added `rpucuda` C++ simulator, exposed through a `pybind` interface.
* Added a PyTorch `AnalogLinear` neural network model.
* Added a PyTorch `AnalogConv2d` neural network model.


[UNRELEASED]: https://github.com/IBM/aihwkit/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/IBM/aihwkit/compare/v0.2.1..v0.3.0
[0.2.1]: https://github.com/IBM/aihwkit/compare/v0.2.0..v0.2.1
[0.2.0]: https://github.com/IBM/aihwkit/compare/v0.1.0..v0.2.0
[0.1.0]: https://github.com/IBM/aihwkit/releases/tag/v0.1.0

[Keep a Changelog]: https://keepachangelog.com/en/1.0.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
