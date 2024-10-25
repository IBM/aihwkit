# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Wrapper for the RPUCuda C++ tiles."""

from typing import Optional, Union, Dict, Tuple, Any

from torch import Tensor, zeros, tensor
from torch import device as torch_device
from torch.nn import Parameter
from torch.cuda import device as cuda_device
from torch.autograd import no_grad

from aihwkit.exceptions import CudaError
from aihwkit.simulator.parameters.base import RPUConfigGeneric
from aihwkit.simulator.rpu_base import cuda, tiles
from aihwkit.simulator.tiles.base import SimulatorTileWrapper
from aihwkit.exceptions import ArgumentError
from aihwkit.optim.context import AnalogContext


if cuda.is_compiled():
    MAP_TILE_CLASS_TO_CUDA = {
        tiles.AnalogTile: tiles.CudaAnalogTile,
        tiles.FloatingPointTile: tiles.CudaFloatingPointTile,
    }
    if hasattr(tiles, "half"):
        MAP_TILE_CLASS_TO_CUDA.update(
            {
                tiles.half.AnalogTile: tiles.half.CudaAnalogTile,
                tiles.half.FloatingPointTile: tiles.half.CudaFloatingPointTile,
            }
        )
    if hasattr(tiles, "double"):
        MAP_TILE_CLASS_TO_CUDA.update(
            {
                tiles.double.AnalogTile: tiles.double.CudaAnalogTile,
                tiles.double.FloatingPointTile: tiles.double.CudaFloatingPointTile,
            }
        )
    if hasattr(tiles, "bfloat16"):
        MAP_TILE_CLASS_TO_CUDA.update(
            {
                tiles.bfloat16.AnalogTile: tiles.bfloat16.CudaAnalogTile,
                tiles.bfloat16.FloatingPointTile: tiles.bfloat16.CudaFloatingPointTile,
            }
        )

else:
    MAP_TILE_CLASS_TO_CUDA = {}


class RPUCudaSimulatorTileWrapper(SimulatorTileWrapper):
    """Wraps the RPUCuda simulator tile.

    This class adds some functionality to the minimalistic
    ``SimulatorTileWrapper`` specific to the RPUCuda tiles that are
    handled in C++ through python bindings .

    Args:
        out_size: output size
        in_size: input size
        rpu_config: resistive processing unit configuration.
        bias: whether to add a bias column to the tile.
        in_trans: Whether to assume an transposed input (batch first)
        out_trans: Whether to assume an transposed output (batch first)
        shared_weights: optional shared weights tensor memory that
            should be used.
    """

    # pylint: disable=abstract-method, too-many-public-methods

    def __init__(
        self,
        out_size: int,
        in_size: int,
        rpu_config: RPUConfigGeneric,
        bias: bool = True,
        in_trans: bool = False,
        out_trans: bool = False,
        shared_weights: bool = False,
    ) -> None:
        SimulatorTileWrapper.__init__(
            self,
            out_size,
            in_size,
            rpu_config,
            bias,
            in_trans,
            out_trans,
            torch_update=False,
            handle_output_bound=True,
        )

        self.shared_weights = None  # type: Parameter
        if shared_weights:
            self.shared_weights = Parameter(
                zeros(out_size, in_size + int(self.analog_bias), dtype=self.get_dtype())
            )
            self.ensure_shared_weights()

    def get_forward_out_bound(self) -> Optional[float]:
        """Helper for getting the output bound to correct the
        gradients using the AnalogFunction.
        """
        if hasattr(self.rpu_config, "forward") and self.rpu_config.forward.out_bound > 0:
            return self.rpu_config.forward.out_scale * self.rpu_config.forward.out_bound * 0.999
        return None

    @no_grad()
    def cpu(self) -> "SimulatorTileWrapper":
        """Return a copy of this tile in CPU memory.

        Returns:
            self in case of CPU
        """
        if not self.is_cuda:
            return self
        super().cpu()
        state_dict = self.__getstate__()
        for value in state_dict.values():
            if isinstance(value, AnalogContext):
                value.data = value.data.cpu()
        self.__setstate__(state_dict)
        return self

    @no_grad()
    def cuda(
        self, device: Optional[Union[torch_device, str, int]] = None
    ) -> "SimulatorTileWrapper":
        """Return a copy of the  tile in CUDA memory.

        Args:
            device: CUDA device

        Returns:
            Self with the underlying C++ tile moved to CUDA memory.

        Raises:
            CudaError: if the library has not been compiled with CUDA.
        """

        if not cuda.is_compiled():
            raise CudaError("aihwkit has not been compiled with CUDA support")

        device = torch_device("cuda", cuda_device(device).idx)

        if self.is_cuda and device != self.device:
            return self.cpu().cuda(device)
        if self.tile.__class__ in MAP_TILE_CLASS_TO_CUDA:
            with cuda_device(device):
                self.tile = MAP_TILE_CLASS_TO_CUDA[self.tile.__class__](self.tile)
                self.is_cuda = True
                self.device = device
                self.analog_ctx.data = self.analog_ctx.data.cuda(device)
                self.analog_ctx.reset(self)  # type: ignore

            if self.shared_weights is not None:
                self.shared_weights.data = zeros(
                    self.tile.get_x_size(),
                    self.tile.get_d_size(),
                    dtype=self.get_dtype(),
                    requires_grad=True,
                ).cuda(device)
                # ensure shared weights will be called later (needs copying still)

        return self

    @no_grad()
    def ensure_shared_weights(self, shared_weights: Optional[Tensor] = None) -> None:
        """Ensure that the shared_weights is set properly.

        Caution:
           This is only called from analog function.

        No-op if shared weights is not used.
        """
        if shared_weights is not None:
            self.shared_weights.data = shared_weights.data  # type: ignore

        if self.shared_weights is not None:
            self.tile.set_shared_weights(self.shared_weights.data)  # type: ignore

    @no_grad()
    def set_delta_weights(self, delta_weights: Optional[Tensor] = None) -> None:
        """Set the weight grad tensor and set the update to.

        No-op if shared weights is not used.
        """
        if self.shared_weights is not None and delta_weights is not None:
            self.tile.set_delta_weights(delta_weights)

    @no_grad()
    def reset_delta_weights(self) -> None:
        """Reset the weight grad tensor to default update behavior (i.e. adding the
        update directly to the weight).

        No-op if shared weights is not used.
        """
        if self.shared_weights is not None:
            self.tile.reset_delta_weights()

    def get_hidden_update_index(self) -> int:
        """Get the current updated device index of the hidden devices.

        Usually this is 0 as only one device is present per
        cross-point for many tile RPU configs. However, some RPU
        configs maintain internally multiple devices per cross-point
        (e.g. :class:`~aihwkit.simulator.config.devices.VectorUnitCell`).

        Returns:
            The next mini-batch updated device index.

        Note:
            Depending on the update and learning policy implemented
            in the tile, updated devices might switch internally as
            well.
        """
        return self.tile.get_hidden_update_index()

    def set_hidden_update_index(self, index: int) -> None:
        """Set the current updated hidden device index.

        Usually this is ignored and fixed to 0 as only one device is
        present per cross-point. Other devices, might not allow
        explicit setting as it would interfere with the implemented
        learning rule. However, some tiles have internally
        multiple devices per cross-point (eg. unit cell) that can be
        chosen depending on the update policy.

        Args:
            index: device index to be updated in the next mini-batch

        Note:
            Depending on the update and learning policy implemented
            in the tile, updated devices might switch internally as
            well.
        """
        self.tile.set_hidden_update_index(index)

    def _get_extra_parameters(
        self, pre_key: str, full_key: bool = False
    ) -> Tuple[Union[Dict[Tuple[str, str], Tensor], Dict[str, Tensor]], Dict[str, Any]]:
        """Get the sub keys in the extra starting with pre_key."""
        extra = self.tile.dump_extra()
        if full_key:
            dic = {
                (key.split(pre_key)[-1], key): tensor(value)
                for key, value in extra.items()
                if pre_key in key
            }
        else:
            dic = {
                key.split(pre_key)[-1]: tensor(value)
                for key, value in extra.items()
                if pre_key in key
            }

        return dic, extra

    def _set_extra_parameters(self, pre_key: str, dic: Dict[str, Any]) -> None:
        """Set the sub keys in the extra starting with pre_key.

        Raises:
            ArgumentError: in case a length mismatch with the stored values exists
        """
        org_dic, extra = self._get_extra_parameters(pre_key, full_key=True)  # type: ignore

        key_lst = []
        for (key, full_key), org_value in org_dic.items():  # type: ignore
            if key not in dic:
                continue
            key_lst.append(key)
            new_value = dic[key].tolist()
            if len(new_value) != len(org_value):
                raise ArgumentError(f"Length mismatch in parameter '{key}'!")
            extra[full_key] = new_value

        if len(set(list(dic.keys())) - set(key_lst)) > 0:
            raise ArgumentError("Some given dict key names do not exist!")

        self.load_extra(extra)

    def get_forward_parameters(self) -> Dict[str, Tensor]:
        """Get the additional parameters generated for the forward pass.

        Returns:
            Dictionary of the forward parameters set.
        """
        return self._get_extra_parameters("fb_pass.fwd.")[0]  # type: ignore

    def set_forward_parameters(
        self, dic: Optional[Dict[str, Tensor]] = None, **kwargs: Dict[str, Tensor]
    ) -> None:
        """Set the additional parameters generated for the forward pass.

        Args:
            dic: dictionary of parameters to set (from :meth:`get_forward_parameter`)
            kwargs: parameter names can alternatively given directly as keywords
        """
        if dic is None:
            dic = kwargs
        return self._set_extra_parameters("fb_pass.fwd.", dic)

    def get_backward_parameters(self) -> Dict[str, Tensor]:
        """Get the additional parameters generated for the backward pass.

        Returns:
            Dictionary of the forward parameters set.
        """
        return self._get_extra_parameters("fb_pass.bwd.")[0]  # type: ignore

    def set_backward_parameters(
        self, dic: Optional[Dict[str, Tensor]], **kwargs: Dict[str, Tensor]
    ) -> None:
        """Set the additional parameters generated for the backward pass.

        Args:
            dic: dictionary of parameters to set (from :meth:`get_backward_parameter`)
            kwargs: parameter names can alternatively given directly as keywords
        """
        if dic is None:
            dic = kwargs
        return self._set_extra_parameters("fb_pass.bwd.", dic)

    def decay_weights(self, alpha: float = 1.0) -> None:
        """Decays the weights once according to the decay parameters of the tile.

        Args:
            alpha: additional decay scale (such as LR). The base decay
                rate is set during tile init.

        Returns:
            None.
        """
        return self.tile.decay_weights(alpha)

    def drift_weights(self, delta_t: float = 1.0) -> None:
        """Drifts the weights once according to the drift parameters of the
        tile.

        See also :class:`~aihwkit.simulator.configs.DriftParameter`.

        Args:
            delta_t: Time since last drift call.

        Returns:
            None.
        """
        return self.tile.drift_weights(delta_t)

    def diffuse_weights(self) -> None:
        """Diffuses the weights once according to the diffusion parameters of
        the tile.

        The base diffusion rate is set during tile init.

        Returns:
            None
        """
        return self.tile.diffuse_weights()

    def reset_columns(
        self, start_column_idx: int = 0, num_columns: int = 1, reset_prob: float = 1.0
    ) -> None:
        r"""Reset (a number of) columns according to the reset parameters of the tile.

        Resets the weights with device-to-device and cycle-to-cycle
        variability (depending on device type), typically:

        .. math::
            W_{ij} = \xi*\sigma_\text{reset} + b^\text{reset}_{ij}

        The reset parameters are set during tile init.

        Args:
            start_column_idx: a start index of columns (0..x_size-1)
            num_columns: how many consecutive columns to reset (with circular warping)
            reset_prob: individual probability of reset.

        Returns:
            None
        """
        return self.tile.reset_columns(start_column_idx, num_columns, reset_prob)

    def reset(self, reset_prob: float = 1.0) -> None:
        r"""Reset the updated device tile according to the reset parameters of the tile.

        Resets the weights with device-to-device and cycle-to-cycle
        variability (depending on device type), typically:

        .. math::
            W_{ij} = \xi*\sigma_\text{reset} + b^\text{reset}_{ij}

        The reset parameters are set during tile init.

        Args:
            reset_prob: individual probability of reset.

        Returns:
            None
        """
        return self.tile.reset_columns(0, -1, reset_prob)

    def set_verbosity_level(self, verbose: int) -> None:
        """Set verbosity level of tile.

        Args:
            verbose: level of verbosity
        """
        self.tile.set_verbosity_level(verbose)

    def dump_extra(self) -> Optional[Dict[str, Any]]:
        """Dumps any extra states / attributed necessary for
        checkpointing.

        For Tiles based on Modules, this should be normally handled by
        torch automatically.
        """
        return self.tile.dump_extra()

    def load_extra(self, extra: Dict[str, Any], strict: bool = False) -> None:
        """Load any extra states / attributed necessary for
        loading from checkpoint.

        For Tiles based on Modules, this should be normally handled by
        torch automatically.

        Note:
            Expects the exact same RPUConfig / device etc for applying
            the states. Cross-loading of state-dicts is not supported
            for extra states, they will be just ignored.

        Args:
            extra: dictionary of states from `dump_extra`.
            strict: Whether to throw an error if keys are not found.
        """
        self.tile.load_extra(extra, strict)

    def post_update_step(self) -> None:
        """Operators that need to be called once per mini-batch.

        Note:
            This function is called by the analog optimizer.

        Caution:
            If no analog optimizer is used, the post update steps will
            not be performed.
        """

        if self.rpu_config.device.requires_diffusion():
            self.tile.diffuse_weights()
        if self.rpu_config.device.requires_decay():
            self.tile.decay_weights()
