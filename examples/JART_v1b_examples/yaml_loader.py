import yaml
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import JARTv1bDevice

def from_yaml(config_file):
    split = config_file.split(".")
    if len(split) == 2:
        job_type = split[0]
    else:
        job_type = config_file

    import yaml
    stream = open(config_file, "r")
    config_dictionary = yaml.safe_load(stream)

    project_name = config_dictionary["project_name"]
    CUDA_Enabled = config_dictionary["USE_CUDA"]
    USE_wandb = config_dictionary["USE_wandb"]
    USE_0_initialization= config_dictionary["USE_0_initialization"]
    USE_bias= False
    del config_dictionary["USE_0_initialization"]
    del config_dictionary["project_name"]
    del config_dictionary["USE_wandb"]

    if "Repeat_Times" in config_dictionary:
        Repeat_Times = config_dictionary["Repeat_Times"]
        del config_dictionary["Repeat_Times"]
    else:
        Repeat_Times = 1

    # Define a single-layer network.
    JART_rpu_config = SingleRPUConfig(device=JARTv1bDevice(w_max=config_dictionary["w_max"],
                                                    w_min=config_dictionary["w_min"],

                                                    read_voltage=config_dictionary["pulse_related"]["read_voltage"],
                                                    pulse_voltage_SET=config_dictionary["pulse_related"]["pulse_voltage_SET"],
                                                    pulse_voltage_RESET=config_dictionary["pulse_related"]["pulse_voltage_RESET"],
                                                    pulse_length=config_dictionary["pulse_related"]["pulse_length"],
                                                    base_time_step=config_dictionary["pulse_related"]["base_time_step"],

                                                    enable_w_max_w_min_bounds=config_dictionary["noise"]["enable_w_max_w_min_bounds"],
                                                    w_max_dtod=config_dictionary["noise"]["w_max"]["device_to_device"],
                                                    w_max_dtod_upper_bound=config_dictionary["noise"]["w_max"]["dtod_upper_bound"],
                                                    w_max_dtod_lower_bound=config_dictionary["noise"]["w_max"]["dtod_lower_bound"],
                                                    w_min_dtod=config_dictionary["noise"]["w_min"]["device_to_device"],
                                                    w_min_dtod_upper_bound=config_dictionary["noise"]["w_min"]["dtod_upper_bound"],
                                                    w_min_dtod_lower_bound=config_dictionary["noise"]["w_min"]["dtod_lower_bound"],

                                                    Ndiscmax_dtod=config_dictionary["noise"]["Ndiscmax"]["device_to_device"],
                                                    Ndiscmax_dtod_upper_bound=config_dictionary["noise"]["Ndiscmax"]["dtod_upper_bound"],
                                                    Ndiscmax_dtod_lower_bound=config_dictionary["noise"]["Ndiscmax"]["dtod_lower_bound"],
                                                    Ndiscmax_std=config_dictionary["noise"]["Ndiscmax"]["cycle_to_cycle_direct"],
                                                    Ndiscmax_ctoc_upper_bound=config_dictionary["noise"]["Ndiscmax"]["ctoc_upper_bound"],
                                                    Ndiscmax_ctoc_lower_bound=config_dictionary["noise"]["Ndiscmax"]["ctoc_lower_bound"],

                                                    Ndiscmin_dtod=config_dictionary["noise"]["Ndiscmin"]["device_to_device"],
                                                    Ndiscmin_dtod_upper_bound=config_dictionary["noise"]["Ndiscmin"]["dtod_upper_bound"],
                                                    Ndiscmin_dtod_lower_bound=config_dictionary["noise"]["Ndiscmin"]["dtod_lower_bound"],
                                                    Ndiscmin_std=config_dictionary["noise"]["Ndiscmin"]["cycle_to_cycle_direct"],
                                                    Ndiscmin_ctoc_upper_bound=config_dictionary["noise"]["Ndiscmin"]["ctoc_upper_bound"],
                                                    Ndiscmin_ctoc_lower_bound=config_dictionary["noise"]["Ndiscmin"]["ctoc_lower_bound"],

                                                    ldisc_dtod=config_dictionary["noise"]["ldisc"]["device_to_device"],
                                                    ldisc_dtod_upper_bound=config_dictionary["noise"]["ldisc"]["dtod_upper_bound"],
                                                    ldisc_dtod_lower_bound=config_dictionary["noise"]["ldisc"]["dtod_lower_bound"],
                                                    ldisc_std=config_dictionary["noise"]["ldisc"]["cycle_to_cycle_direct"],
                                                    ldisc_std_slope=config_dictionary["noise"]["ldisc"]["cycle_to_cycle_slope"],
                                                    ldisc_ctoc_upper_bound=config_dictionary["noise"]["ldisc"]["ctoc_upper_bound"],
                                                    ldisc_ctoc_lower_bound=config_dictionary["noise"]["ldisc"]["ctoc_lower_bound"],

                                                    rdisc_dtod=config_dictionary["noise"]["rdisc"]["device_to_device"],
                                                    rdisc_dtod_upper_bound=config_dictionary["noise"]["rdisc"]["dtod_upper_bound"],
                                                    rdisc_dtod_lower_bound=config_dictionary["noise"]["rdisc"]["dtod_lower_bound"],
                                                    rdisc_std=config_dictionary["noise"]["rdisc"]["cycle_to_cycle_direct"],
                                                    rdisc_std_slope=config_dictionary["noise"]["rdisc"]["cycle_to_cycle_slope"],
                                                    rdisc_ctoc_upper_bound=config_dictionary["noise"]["rdisc"]["ctoc_upper_bound"],
                                                    rdisc_ctoc_lower_bound=config_dictionary["noise"]["rdisc"]["ctoc_lower_bound"]))
        
    return job_type, project_name, CUDA_Enabled, USE_wandb, USE_0_initialization, USE_bias, Repeat_Times, config_dictionary, JART_rpu_config
    