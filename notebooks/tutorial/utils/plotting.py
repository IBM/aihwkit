from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams["font.size"] = 16
rcParams["axes.linewidth"] = 1.1
rcParams["axes.labelpad"] = 3.0
plot_color_cycle = plt.cycler(
    "color",
    [
        "#9b59b6",
        "#3498db",
        "#95a5a6",
        "#e74c3c",
        "#34495e",
        "#2ecc71",
        "#1E2460",
        "#B5B8B1",
        "#734222",
        "#A52019",
    ],
)
rcParams["axes.prop_cycle"] = plot_color_cycle
rcParams["axes.xmargin"] = 0
rcParams["axes.ymargin"] = 0
rcParams.update(
    {
        "figure.figsize": (6.4, 4.8),
        "figure.subplot.left": 0.07,
        "figure.subplot.right": 0.946,
        "figure.subplot.bottom": 0.1,
        "figure.subplot.top": 0.965,
        "axes.autolimit_mode": "round_numbers",
        "axes.grid": True,
        "xtick.major.size": 7,
        "xtick.minor.size": 3.5,
        "xtick.major.width": 1.1,
        "xtick.minor.width": 1.1,
        "xtick.major.pad": 5,
        "xtick.minor.visible": True,
        "ytick.major.size": 7,
        "ytick.minor.size": 3.5,
        "ytick.major.width": 1.1,
        "ytick.minor.width": 1.1,
        "ytick.major.pad": 5,
        "ytick.minor.visible": True,
        "lines.markersize": 10,
        "lines.markerfacecolor": "none",
        "lines.markeredgewidth": 0.8,
    }
)
