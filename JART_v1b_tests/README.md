# JART v1b Simulator Test Scripts
This folder contains the code producing the figures of the paper "Integration of Physics Derived Memristor Models with Machine Learning Frameworks". At the moment the GPU implementation still have some issues, so the CPU version is used to generate the results shown in the paper.

Different configuration sets are discribed with the .yml files, and it's easy to costimize configurations for your own purpose.

[WandB](https://wandb.ai/site) was used to record the data during training.

## Basic Tests
Some basic test was provided to debug the setup. This includes:

Linear Regression with a single neuron:

```
python 00_basic_test_v1b.py -c [config_file.yml]
```

Simple network with one fully connected layer:

```
python 01_simple_layer_v1b.py -c [config_file.yml]
```

Simple network with 3 fully connected layers:

```
python 02_multiple_layer_v1b.py -c [config_file.yml]
```

## MINST Tests
We tested the device model with the MNIST dataset. 
A scheduler that decrease the learning rate by 50 percent every 10 epochs is used with the model. This is needed because the switching speed of our memristor model varies a lot at different conductance range. The network will become unstable and start to oscillate without decreasing learning rate through out the training course.

Test the network using JART v1b device model with:

```
python 03_mnist_training_v1b.py -c [config_file.yml]
```

Test the network using floating point weights with:

```
python mnist_training_floating_point.py
```