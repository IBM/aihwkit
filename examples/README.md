# IBM Analog Hardware Acceleration Kit: Extended Examples 

## Example 24: [`24_bert_on_squad.py`]

This example is adapted from
https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb

The example loads a pre-trained BERT model trained on
the SQuAD dataset. It then applies `convert_to_analog()`
to examine the effects of `drift_analog_weights()` on inference performance at
different weight noise levels. Tensorboard is used to display the SQuAD
metrics evaluated using the model at various times after training completed.

Commandline arguments can be used to control certain options.
For example:
`python /path/to/aihwkit/examples/24_bert_on_squad.py -n 0.1 -r "run 1" -l 0.0005 -t`
to set the weight noise to 0.1, name the run in Tensorboard "run 1",
set the learning rate to 0.0005, and do hardware-aware training

## Example 31: ['31_gpt2_on_openwebtext.py']
This example is adapted from
https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb

The example loads a pre-trained GPT-2 model trained on
the openwebtext dataset. It then applies convert_to_analog()
to examine the effects of drift_analog_weights() on inference performance at
different weight noise levels. Tensorboard is used to display the perplexity
metrics evaluated using the model at various times after training completed.

Commandline arguments can be used to control certain options. For example:
python /path/to/aihwkit/examples/31_gpt2_on_openwebtext.py -n 0.1 -r "run 1" -l 0.0005 -t
to set the weight noise to 0.1, name the run in Tensorboard "run 1",
set the learning rate to 0.0005, and do hardware-aware training.

## Example 32: ['32_bert_on_squad_noise_analysis.py']
This example is adapted from
https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb

The example loads a pre-trained BERT model trained on
the SQuAD dataset. It then applies convert_to_analog()
to examine the effects of noise on different layers of the model. Specifically, it evaluates the sensitivity of each layer to noise by injecting noise into individual layers and measuring the resulting drop in F1 score and exact match metrics on the SQuAD task.

The script uses Tensorboard to display the SQuAD metrics evaluated at various times after training completed, and command-line arguments can be used to control various options. For example:
python /path/to/aihwkit/examples/32_bert_noise_analysis.py -n 0.1 -r "run 1" -l 0.0005 -t
to set the weight noise to 0.1, name the run in Tensorboard "run 1", set the learning rate to 0.0005, and enable hardware-aware training.

Additionally, the script allows for the loading of a pre-trained analog model from a checkpoint to evaluate the noise sensitivity of different layers, providing insights into which layers are most vulnerable to noise and thus require more robust analog hardware implementations.

- Train and save the analog model: python 32_bert_on_squad_noise_analysis.py --noise 0.1 --train_hwa --checkpoint ./saved_chkpt_noise_analysis.pth
- Load and evaluate the analog model: python 32_bert_on_squad_noise_analysis.py --noise 0.1 --checkpoint ./saved_chkpt_noise_analysis.pth --load

#Major Changes
1. Identify the Layers: Extract the names or indices of each layer in the model.
2. Modify the Noise Configuration: Implement a mechanism to apply noise only to one layer at a time while keeping other layers noise-free.
3. Evaluate the Impact: Evaluate the impact of noise on each layer by measuring the F1 score drop for each noisy layer configuration.

apply_noise_to_layer: This function applies noise to a specific layer in the model.
get_all_layers: This function returns the names of all layers in the model.
evaluate_noise_sensitivity: This function evaluates the noise sensitivity for each layer by applying noise to one layer at a time, performing inference, and measuring the F1 score and exact match metrics.
main: Updated to perform noise sensitivity analysis by calling the new functions.

[`01_simple_layer.py`]: 01_simple_layer.py
[`02_multiple_layer.py`]: 02_multiple_layer.py
[`03_minst_training.py`]: 03_minst_training.py
[`04_lenet5_training.py`]: 04_lenet5_training.py
[`05_simple_layer_hardware_aware.py`]: 05_simple_layer_hardware_aware.py
[`06_lenet5_hardware_aware.py`]: 06_lenet5_hardware_aware.py
[`07_simple_layer_with_other_devices.py`]: 07_simple_layer_with_other_devices.py
[`08_simple_layer_with_tiki_taka.py`]: 08_simple_layer_with_tiki_taka.py
[`09_simple_layer_deterministic_pulses.py`]: 09_simple_layer_deterministic_pulses.py
[`10_plot_presets.py`]: 10_plot_presets.py
[`11_vgg8_training.py`]: 11_vgg8_training.py
[`12_simple_layer_with_mixed_precision.py`]: 12_simple_layer_with_mixed_precision.py
[`13_experiment_3fc.py`]: 13_experiment_3fc.py
[`14_experiment_custom_scheduler.py`]: 14_experiment_custom_scheduler.py
[`15_simple_lstm.py`]: 15_simple_lstm.py
[`16_mnist_gan.py`]: 16_mnist_gan.py
[`17_resnet34_digital_to_analog.py`]: 17_resnet34_imagenet_conversion_to_analog.py
[`18_cifar10_on_resnet.py`]: 18_cifar10_on_resnet.py
[`19_analog_summary_lenet.py`]: 19_analog_summary_lenet.py
[`20_mnist_ddp.py`]: 20_mnist_ddp.py
[`21_fit_device_data.py`]: 21_fit_device_data.py
[`22_war_and_peace_lstm.py`]: 22_war_and_peace_lstm.py
[`23_using_analog_tile_as_matrix.py`]: 23_using_analog_tile_as_matrix.py
[`24_bert_on_squad.py`]: 24_bert_on_squad.py
[`25_torch_tile_lenet5_hardware_aware.py`]: 25_torch_tile_lenet5_hardware_aware.py
[`26_correlation_detection.py`]: 26_correlation_detection.py
[`27_input_range_calibration`]: 27_input_range_calibration.py
[`28_advanced_irdrop.py`]: 28_advanced_irdrop.py
[`29_linalg_krylov.py`]: 29_linalg_krylov.py
[`30_external_hardware_aware_model.py`]: 30_external_hardware_aware_model.py
