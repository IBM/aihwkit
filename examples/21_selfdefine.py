import matplotlib.pyplot as plt
from aihwkit.utils.visualization import plot_device_compact
from aihwkit.simulator.configs.devices import SelfDefineDevice

# define up/down pulse vs weight relationship
# n number of points
up_pulse = [0.001, 0.002, 0.003, 0.004, 0.005]
up_weight = [1.0, 0.5, 0.0, -0.5, -1.0]

down_pulse = [0.001, 0.002, 0.003, 0.004, 0.005]
down_weight = [-1.0, -0.5, 0.0, 0.5, 1.0]
n_points = 5

plt.ion()
plot_device_compact(
    SelfDefineDevice(w_min=-1, w_max=1, dw_min=0.01, pow_gamma=1.1, pow_gamma_dtod=0.0,
                  pow_up_down=0.2, w_min_dtod=0.0, w_max_dtod=0.0, def_up_pulse=up_pulse, 
                                                                   def_up_weight=up_weight, 
                                                                   def_down_pulse=down_pulse, 
                                                                   def_down_weight=down_weight,
                                                                   def_n_points=n_points), n_steps=1000)
plt.show()
plt.savefig('my_figure.png')