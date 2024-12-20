# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Visualization utilities (web)."""

import argparse
from typing import Optional, Union
from pathlib import Path

from cycler import cycler
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from aihwkit.simulator.presets.devices import (
    ReRamSBPresetDevice,
    ReRamESPresetDevice,
    CapacitorPresetDevice,
    EcRamPresetDevice,
    IdealizedPresetDevice,
)
from aihwkit.simulator.presets.compounds import PCMPresetUnitCell
from aihwkit.simulator.configs.devices import PulsedDevice
from aihwkit.utils.visualization import plot_device_compact

# Colors used by the frontend.
WEB_COLORS = [
    "#8A3FFC",
    "#33B1FF",
    "#007D79",
    "#FF7EB6",
    "#FA4D56",
    "#FFF1F1",
    "#6FDC8C",
    "#4589FF",
    "#D12771",
    "#D2A106",
    "#08BDBA",
    "#BAE6FF",
    "#BA4E00",
    "#D4BBFF",
]
# Devices for which plots should be generated.
DEVICES = {
    ReRamESPresetDevice: 1000,
    ReRamSBPresetDevice: 1000,
    CapacitorPresetDevice: 400,
    EcRamPresetDevice: 1000,
    IdealizedPresetDevice: 10000,
    PCMPresetUnitCell: 80,
}


def set_dark_style(axes: Axes) -> None:
    """Sets a nice color cycle for a given axes."""

    axes.set_prop_cycle(cycler(color=WEB_COLORS))

    axes.set_facecolor("#262626")


def plot_device_compact_web(
    device: PulsedDevice, w_noise: float = 0.0, n_steps: Optional[int] = None, n_traces: int = 3
) -> Union[Figure, Axes]:
    """Plots a compact step response figure for a given device (preset).

    Note:
        It will use an amount of read weight noise ``w_noise`` for
        reading the weights.

    Params:
        device: PulsedDevice parameters
        w_noise: Weight noise standard deviation during read
        n_steps: Number of steps for up/down cycle
        n_traces: Number of traces to plot (for device-to-device variation)
        show: if `True`, displays the figure.

    Returns:
        the compact step response figure.
    """
    plt.style.use("dark_background")

    figure = plot_device_compact(device, w_noise, n_steps, n_traces)

    if isinstance(figure, Axes):
        return figure

    # Tune for web.
    for axes in figure.get_axes():
        for i, line in enumerate(axes.get_lines()):
            line.set_color(WEB_COLORS[i])
        # set_dark_style(axes)

    return figure


def save_plots_for_web(path: Path = Path("/tmp"), file_format: str = "svg") -> None:
    """Create the plots for the web.

    Args:
        path: the path where the images will be stored.
        file_format: the image format.
    """

    def camel_to_snake(source: str) -> str:
        """Convert a CamelCase string into snake-case."""
        return "".join(["_" + char.lower() if char.isupper() else char for char in source]).lstrip(
            "_"
        )

    for device, n_steps in DEVICES.items():
        # Images for the detailed modal.
        file_name = "{}.{}".format(camel_to_snake(device.__name__), file_format)
        file_path = path.absolute() / file_name

        figure = plot_device_compact_web(device(), n_steps=n_steps)  # type: ignore

        figure.savefig(  # type: ignore
            file_path, format=file_format, transparent=True, bbox_inches="tight"
        )

        # Images for the mini leftbar.
        file_name = "{}-mini.{}".format(camel_to_snake(device.__name__), file_format)
        file_path = path.absolute() / file_name

        figure = plot_device_compact_web(device(), n_traces=1, n_steps=n_steps)
        for axes in figure.get_axes():  # type: ignore
            # Disable texts.
            axes.set_title("")
            axes.set_xlabel("")
            axes.set_ylabel("")
            # Disable tick labels.
            axes.xaxis.set_ticklabels([])
            axes.yaxis.set_ticklabels([])
            # Disable axis entirely.
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)
            # Increase axis width.
            for axis in ["top", "bottom", "left", "right"]:
                axes.spines[axis].set_linewidth(3)
            for line in axes.get_lines():
                line.set_linewidth(4)
        figure.savefig(  # type: ignore
            file_path, format=file_format, transparent=True, bbox_inches="tight"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for frontend.")
    parser.add_argument("destination", type=str, help="folder where the plots will be stored")
    args = parser.parse_args()

    destination_path = Path(args.destination)
    destination_path.mkdir(exist_ok=True)
    save_plots_for_web(destination_path)
