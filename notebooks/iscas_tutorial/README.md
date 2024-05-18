## Interactive Task
Deployment of a HWA Trained Network to Fusion:
1. Go to https://aihw-composer.draco.res.ibm.com and create an account.
2. Create a New experiment -> Inference experiment -> Real Chip with Program your own Neural Network Weights.
3. Download the ResNet-9 pretrained network checkpoint: https://aihwkit-tutorial.s3.us-east.cloud-object-storage.appdomain.cloud/resnet9s.th.
4. Load the example Jupyter Notebook from the website https://github.com/IBM/aihwkit/blob/master/notebooks/analog_fusion.ipynb.
5. Modify cell [5] to return the ResNet-9 network model definition from example 30: https://github.com/IBM/aihwkit/blob/master/examples/30_external_hardware_aware_model.py.
6. Modify cell [8] to load CIFAR-10 test dataset from torchvision datasets (code from example 30).
7. Modify cell [11] to prepare the ResNet-9 model by loading the checkpoint state dictionaries and convert it to analog (code from example 30).
8. Create the 'resnet9s_target_weights.csv' file of conductance values to be uploaded on the composer website.
9. Program the conductance values on Fusion chip using the analog composer, run the experiment, and download the resulting csv file of programmed conductance values
10. Finally, use the fusion_import utility to set the model weights to the conductance values read from the chip and evaluate the model.
