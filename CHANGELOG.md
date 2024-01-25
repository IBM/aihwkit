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

## Unreleased

## [0.9.0] - 2024/01/25

### Added

* On-the-fly change of some `RPUConfig` fields (\# 539)
* Fusion chip CSV file model weights exporter functionality (\#538)
* Experimental support for RPU data types (\#563)
* Optional AIHWKIT C++ extension module (\#563)
* Variable mantissa / exponent tensor conversion operator (\#563)
* To digital feature for analog layers (\#563)
* New `PCM_NOISE` type for hardware-aware training for inference (\#563)
* Transfer compounds using torch implementation (`TorchTransferTile`) (\#567)
* Weight programming error plotting utility (\#572)
* Add optimizer checkpoint in example 20 (\#573)
* Inference tile with time-dependent IR-drop (\#587)
* Linear algebra module (\#588)
* New Jupyter notebook for Fusion chip access (\#601)

### Fixed

* Repeated call of `cuda()` reset the weights for `InferenceTile` (\#540)
* Custom tile bugfixes (\#563)
* Bug-fixes for specialized learning algorithms (\#563)
* Bug-fix for data-parallel hardware-aware training for inference (\#569)
* Fix docker build stubgen (\#581)
* Fix readthedoc builds (\#586)
* Fix the backward of the input ranges in the torch tile (\#606)

### Changed

* Parameter structure changed into separate files to reduce file sizes (\#563)
* `RPUConfig` has a new `runtime` field and inherits from additional base
  classes (\#563)
* `AnalogWrapper` now directly adds module classes to subclasses (\#563)
* RNN linear layers more custonable (\#563)
* Parameters for specialized learning algorithms changed somwhat (\#563)
* RNN modules inherit from `Module` or `AnalogContainerBase` instead of `AnalogSequential` (\#563)
* Adjustment of parameter to bindings for various number formats (\#563)
* Documentation updates and fixes (\#562, \#564, \#570, \#575, \#576, #\580, #\585, \#586)
* Updated installation instructions in Readthedoc (\#594)

### Removed

## [0.8.0] - 2023/07/14

### Added

* Added new tutorial notebooks to cover the concepts of training,
 hardware-aware training, post-training calibration, and extending aihwkit functionality (\#518, \#523, \#526)
* Calibration of input ranges for inference (\#512)
* New analog in-memory training algorithms: Chopped Tiki-taka II (\#512)
* New analog in-memory training algorithms: AGAD (\#512)
* New training presets: `ReRamArrayOMPresetDevice`,
  `ReRamArrayHfO2PresetDevice`, `ChoppedTTv2*`, `AGAD*` (\#512)
* New correlation detection example for comparing specialized analog SGD
  algorithms (\#512)
* Simplified `build_rpu_config` script for generating `RPUConfigs` for
  analog in-memory SGD (\#512)
* `CustomTile` for customization of in-memory training algorithms (\#512)
* Pulse counters for pulsed analog training (\#512)
* `TorchInferenceTile` for a fully torch-based analog tile for
  inference (not using the C++ RPUCuda engine), supporting a subset of MVM nonidealities (\#512)
* New inference preset `StandardHWATrainingPreset` (\#512)
* New inference noise model `ReRamWan2022NoiseModel` (\#512)
* Improved HWA-training for inference featuring input and output range
  learning and more (\#512)
* Improved CUDA memory management (using torch cached GPU memory for
  internal RPUCuda buffer) (\#512)
* New layer generator: `analog_layers()` loops over layer modules (except
  container) (\#512)
* `AnalogWrapper` for wrapping a full torch module (Without using
  `AnalogSequential`) (\#512)
* `convert_to_digital` utility (\# 512)
* `TileModuleArray` for logical weight matrices larger than a single tile. (\#512)
* Dumping of all C++ fields for accurate analog training saving and
  training continuation after checkpoint load. (\#512)
* `apply_write_noise_on_set` for pulsed devices. (\#512)
* Reset device now also for simple devices. (\#512)
* `SoftBoundsReference`, `PowStepReference` for explicit reference
  subtraction of symmetry point in Tiki-taka (\#512)
* Analog MVM with output-to-output std-deviation variability
  (`output_noise_std`) (\#512)
* Plotting utility for weight errors (\#512)
* `per_batch_sample` weight noise injections for `TorchInferenceRPUConfig` (\#512)


### Fixed

* BERT example 24 using `AnalogWrapper` (\#514)
* Cuda supported testing in examples based on AIHWKIT compilation (\#513)
* Fixed compilation error for CUDA 12.1. (\#500)
* Realistic read weights could have applied the scales wrongly (\#512)


### Changed

* Major re-organization of `AnalogTiles` for increased modularity
  (`TileWithPeriphery`, `SimulatorTile`, `SimulatorTileWrapper`). Analog
 tile modules (possibly array of analog tiles) are now also torch `Module`. (\#512)
* Change in tile generators: `analog_model.analog_tiles()` now loops over
  all available tiles (in all modules) (\#512)
* Import and file position changes. However, user can still import `RPUConfig`
  related modules from `aihwkit.simulator.config` (\#512)
* `convert_to_analog` now also considered mapping. Set
  `mapping.max_out_size = 0` and `mapping.max_out_size = 0` to avoid this. (\#512)
* Mapped layers now use `TileModuleArray` array by default. (\#512)
* Checkpoint structure is different than previous
  versions. `utils.legacy_load` provides a way to load old checkpoints. (\#512)


### Removed

* `realistic_read_write` is removed from some high-level function. Use
  `program_weights` (after setting the weights) or `read_weights`
  for realistic reading (using weight estimation technique).  (\#512)


## [0.7.1] - 2023/03/24

### Added

* Updated the CLI Cloud runner code to support inference experiment result. (\#491)
* Read weights is done with least-square estimation method. (\#489)

### Fixed

* Realistic read / write behavior was broken for some tiles. (\#489)

### Changed

* Torch minimal version has changed to version 1.9. (\#489)
* Realistic read / write is now achieved by `read_weights` and
  `program_weights`. (\#489)

### Removed

* The tile methods `get/set_weights_realistic` are removed. (\#489)


## [0.7.0] - 2023/01/04

### Added
* Reset tiles method (\#456)
* Added many new analog MAC non-linearties (forward / backward pass). (\#456)
* Polynomial weight noise for hardware-aware training. (\#456)
* Remap functionality for hardware-aware training. (\#456)
* Input range estimation for InferenceRPUConfig. (\#456)
* CUDA always syncs and added non-blocking option if not wished. (\#456)
* Fitting utility for fitting any device model to conductance measurements. (\#456)
* Added ``PowStepReferenceDevice`` for easy subtraction of symmetry
  point. (\#456)
* Added ``SoftBoundsReferenceDevice`` for easy subtraction of symmetry
  point. (\#456)
* Added stand-alone functions for applying inference drift to any model. (\#419)
* Added Example 24: analog inference and hardware-aware training on BERT with the SQUAD task. (\#440)
* Added Example 23: how to use ``AnalogTile`` directly to implement an
  analog matrix-vector product without using pytorch modules. (\#393)
* Added Example 22: 2 layer LSTM network trained on War and Peace dataset. (\#391)
* Added a new notebook for exploring analog sensitivities. (\#380)
* Remapping functionality for ``InferenceRPUConfig``. (\#388)
* Inference cloud experiment and runners. (\#410)
* Added ``analog_modules`` generator in ``AnalogSequential``. (\#410)
* Added ``SKIP_CUDA_TESTS`` to manually switch off the CUDA tests.
* Enabling comparisons of ``RPUConfig`` instances. (\#410)
* Specific user-defined function for layer-wise setting for RPUConfigs
  in conversions. (\#412)
* Added stochastic rounding options for ``MixedPrecisionCompound``. (\#418)
* New `remap` parameter field and functionality in
  ``InferenceRPUConfig`` (\#423).
* Tile-level weight getter and setter have `apply_weight_scaling`
  argument. (\#423)
* Pre and post-update / backward / forward methods in `BaseTile` for
  easier user-defined modification of pre and/or post-processings of a tile. (\#423)
* Type-checking for `RPUConfig` fields. (\#424)

### Fixed

* Decay fix for compound devices. (\#463)
* ``RPUCuda`` backend update with many fixes. (\#456)
* Missing zero-grad call in example 02. (\#446)
* Indexing error in ``OneSidedDevice`` for CPU. (\#447)
* Analog summary error when model is on cuda device. (\#392)
* Index error when loading the state dict with a model use previously. (\#387)
* Weights that were not contiguous could have been set wrongly. (\#388)
* Programming noise would not be applied if drift compensation was not
  used. (\#389)
* Loading a new model state dict for inference does not overwrite the noise
  model setting. (\#410)
* Avoid ``AnalogContext`` copying of self pointers. (\#410)
* Fix issue that drift compensation is not applied to conv-layers. (\#412)
* Fix issue that noise modifiers are not applied to conv-layers. (\#412)
* The CPU ``AnalogConv2d`` layer now uses unfolded convolutions instead of
  indexed covolutions (that are efficient only for GPUs). (\#415)
* Fix issue that write noise hidden weights are not transferred to
  pytorch when using ``get_hidden_parameters`` in case of CUDA. (\#417)
* Learning rate scaling due to output scales. (\#423)
* `WeightModifiers` of the `InferenceRPUConfig` are no longer called
  in the forward pass, but instead in the `post_update_step`
  method to avoid issues with repeated forward calls. (\#423)
* Fix training `learn_out_scales` issue after checkpoint load. (\#434)

### Changed

* Pylint / mypy / pycodestyle / protobuf version bump (\#456)
* All configs related classes can now be imported from
  ``aihwkit.simulator.config``. (\#456)
* Weight noise visualization now shows the programming noise and drift
  noise differences. (\#389)
* Concatenate the gradients before applying to the tile update
  function (some speedup for CUDA expected). (\#390)
* Drift compensation uses eye instead of ones for readout. (\#412)
* `weight_scaling_omega_columnwise` parameter in `MappingParameter` is now called
  `weight_scaling_columnwise`. (\#423)
* Tile-level weight getter and setter now use Tensors instead of numpy
  arrays. (\#423)
* Output scaling and mapping scales are now distiniguished, only the
  former is learnable. (\#423)
* Renamed `learn_out_scaling_alpha` parameter in `MappingParameter` to
  `learn_out_scaling` and columnwise learning has a separate switch
  `out_scaling_columnwise`. (\#423)

### Deprecated

* Input `weight_scaling_omega` argument in analog layers is deprecated. (\#423)

### Removed

* The `_scaled` versions of the weight getter and setter methods are
  removed. (\#423)


## [0.6.0] - 2022/05/16

### Added

* Set weights can be used to re-apply the weight scaling omega. (\#360)
* Out scaling factors can be learnt even if weight scaling omega was set to 0. (\#360)
* Reverse up / down option for ``LinearStepDevice``. (\#361)
* Generic Analog RNN classes (LSTM, RNN, GRU) uni or bidirectional. (\#358)
* Added new ``PiecewiseStepDevice`` where the update-step response
  function can be arbitrarily defined by the user in a piece-wise
  linear manner. It can be conveniently used to fit any experimental
  device data. (\#356)
* Several enhancements to the public documentations: added a new
  section for hw-aware training, refreshed the reference API doc, and
  added the newly supported LSTM layers and the mapped conv
  layers. (\#374)

### Fixed

* Legacy checkpoint load with alpha scaling. (\#360)
* Re-application of weight scaling omega when loading checkpoints. (\#360)
* Write noise was not correctly applied for CUDA if ``dw_min_std=0``. (\#356)

### Changed

* The ``set_alpha_scale`` and ``get_alpha_scale`` methods of the C++ tiles are removed. (\#360)
* The lowest supported Python version is now `3.7`, as `3.6` has reached
  end-of-life. Additionally, the library now officially supports Python
  `3.10`. (\#368)


## [0.5.1] - 2022/01/27

### Added

* Load model state dict into a new model with modified `RPUConfig`. (\#276)
* Visualization for noise models for analog inference hardware simulation. (\#278)
* State independent inference noise model. (\# 284)
* Transfer LR parameter for ``MixedPrecisionCompound``. (\#283)
* The bias term can now be handled either by the analog or digital domain by controlling
  the `digital_bias` layer parameter. (\#307)
* PCM short-term weight noise. (\#312)
* IR-drop simulation across columns during analog mat-vec. (\#312)
* Transposed-read for ``TransferCompound``. (\#312)
* ``BufferedTranferCompound`` and TTv2 presets. (\#318)
* Stochastic rounding for ``MixedPrecisionCompound``. (\#318)
* Decay with arbitrary decay point (to reset bias). (\#319)
* Linear layer ``AnalogLinearMapped`` which maps a large weight
  matrix onto multiple analog tiles. (\#320)
* Convolution layers ``AnalogConvNdMapped`` which maps large weight matrix
  onto multiple tiles if necessary. (\#331)
* In the new ``mapping`` field of ``RPUConfig`` the max tile input and
  output sizes can be configured for the ``*Mapped`` layers. (\#331)
* Notebooks directory with several notebook examples (#333, \#334)
* Analog information summary function. (\#316)
* The `alpha` weight scaling factor can now be defined as learnable parameter by switching
  `learn_out_scaling_alpha` in the `rpu_config.mapping` parameters. (\#353)

### Fixed

* Removed GPU warning during destruction when using multiple GPUs. (\#277)
* Fixed issue in transfer counter for mixed precision in case of GPU. (\#283)
* Map location keyword for load / save observed. (\#293)
* Fixed issue with CUDA buffer allocation when batch size changed. (\#294)
* Fixed missing load statedict for ``AnalogSequential``. (\#295)
* Fixed issue with hierarchical hidden parameter settings. (\#313)
* Fixed serious issue that loaded model would not update analog gradients. (\#320)
* Fixed cuda import in examples. (\#320)

### Changed

* The inference noise models are now located in `aihwkit.inference`. (\#281)
* Analog state dict structure `has changed (shared weight are not saved). (\#293)
* Some of the parameter names of the``TransferCompound`` have
  changed. (\#312)
* New fast learning rate parameter for TransferCompound, SGD learning
  rate then is applied on the slow matrix (\#312).
* The ``fixed_value`` of ``WeightClipParameter`` is now  applied for all clipping
  types if set larger than zero. (\#318)
* The use of generators for analog tiles of an ``AnalogModuleBase``. (\#320)
* Digital bias is now accessable through ``MappingParameter``. (\#331)
* The aihwkit documentation. New content around analog ai concepts, training presets, analog ai
  optimizers, new references, and examples. (\#348)
* The `weight_scaling_omega` can now be defined in the `rpu_config.mapping`. (\#353)

### Deprecated

* The module `aihwkit.simulator.noise_models` has been depreciated in
  favor of `aihwkit.inference`. (\#281)


## [0.4.0] - 2021/06/25

### Added

* A number of new config presets added to the library, namely `EcRamMOPreset`,
  `EcRamMO2Preset`, `EcRamMO4Preset`, `TikiTakaEcRamMOPreset`,
  `MixedPrecisionEcRamMOPreset`. These can be used for tile configuration
  (`rpu_config`). They specify a particular device and optimizer choice. (\#207)
* Weight refresh mechanism for `OneSidedUnitCell` to counteract saturation, by
  differential read, reset, and re-write. (\#209)
* Complex cycle-to-cycle noise for `ExpStepDevice`. (\#226)
* Added the following presets: `PCMPresetDevice` (uni-directional),
  `PCMPresetUnitCell` (a pair of uni-directional devices with periodical
  refresh) and a `MixedPrecisionPCMPreset` for using the mixed precision
  optimizer with a PCM pair. (\#226)
* `AnalogLinear` layer now accepts multi-dimensional inputs in the same
  way as PyTorch's `Linear` layer does. (\#227)
* A new `AnalogLSTM` module: a recurrent neural network that uses
 `AnalogLinear`. (\#240)
* Return of weight gradients for `InferenceTile` (only), so that the gradient
  can be handled with any PyTorch optimizer. (\#241)
* Added a generic analog optimizer `AnalogOptimizer` that allows extending
  any existing optimizer with analog-specific features. (\#242)
* Conversion tools for converting torch models into a model having analog
  layers. (\#265)

### Changed

* Renamed the `DifferenceUnitCell` to `OneSidedUnitCell` which more properly
  reflects its function. (\#209)
* The `BaseTile` subclass that is instantiated in the analog layers is now
  retrieved from the new `RPUConfig.tile_class` attribute, facilitating the
  use of custom tiles. (\#218)
* The default parameter for the `dataset` constructor used by `BasicTraining`
  is now the `train=bool` argument. If using a dataset that requires other
  arguments or transforms, they can now be specified via overriding
  `get_dataset_arguments()` and `get_dataset_transform()`. (\#225)
* `AnalogContext` is introduced, along with tile registration function to
  handle arbitrary optimizers, so that re-grouping param groups becomes
  unnecessary. (\#241)
* The `AnalogSGD` optimizer is now implemented based on the generic analog
  optimizer, and its base module is `aihwkit.optim.analog_optimizer`. (\#242)
* The default refresh rate is changed to once per mini-batch for `PCMPreset`
  (as opposed to once per mat-vec). (\#243)

### Deprecated

* Deprecated the `CudaAnalogTile` and `CudaInferenceTile` and
  `CudaFloatingPointTile`. Now the `AnalogTile` can be either on cuda or on cpu
  (determined by the `tile` and the `device` attribute) similar to a torch
  `Tensor`. In particular, call of `cuda()` does not change the `AnalogTile` to
  `CudaAnalogTile` anymore, but only changes the instance in the `tile` field,
  which makes in-place calls to `cuda()` possible. (\#257)

### Removed

* Removed `weight` and `bias` of analog layers from the module parameters as
  these parameters are handled internally for analog tiles. (\#241)

### Fixed

* Fixed autograd functionality for recurrent neural networks. (\#240)
* N-D support for `AnalogLinear`. (\#227)
* Fixed an issue in the `Experiments` that was causing the epoch training loss
  to be higher than the epoch validation loss. (\#238)
* Fixed "Wrong device ordinal" errors for CUDA which resulted from a known
  issue of using CUB together with pytorch. (\#250)
* Renamed persistent weight hidden parameter field to `persistent_weights`.
  (\#251)
* Analog tiles now always move correctly to CUDA when `model.cuda()`
  or `model.to(device)` is used. (\#252, \#257)
* Added an error message when wrong tile class is used for loading an analog
  state dict. (\#262)
* Fixed `MixedPrecisionCompound` being bypassed with floating point compute.
  (\#263)

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


[UNRELEASED]: https://github.com/IBM/aihwkit/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/IBM/aihwkit/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/IBM/aihwkit/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/IBM/aihwkit/compare/v0.7.0..v0.7.1
[0.7.0]: https://github.com/IBM/aihwkit/compare/0.6.0..v0.7.0
[0.6.0]: https://github.com/IBM/aihwkit/compare/v0.5.1..0.6.0
[0.5.1]: https://github.com/IBM/aihwkit/compare/v0.4.0..v0.5.1
[0.4.0]: https://github.com/IBM/aihwkit/compare/v0.3.0..v0.4.0
[0.3.0]: https://github.com/IBM/aihwkit/compare/v0.2.1..v0.3.0
[0.2.1]: https://github.com/IBM/aihwkit/compare/v0.2.0..v0.2.1
[0.2.0]: https://github.com/IBM/aihwkit/compare/v0.1.0..v0.2.0
[0.1.0]: https://github.com/IBM/aihwkit/releases/tag/v0.1.0

[Keep a Changelog]: https://keepachangelog.com/en/1.0.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
