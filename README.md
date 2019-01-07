# Neural Networks from scratch
Yet another python module for implementing neural networks in pure numpy.

## Aim
To develop a python module for training neural networks and using them for inference. The focus is on implementing the necessary methods using only numpy, while being capable of training CNNs efficiently.

## Feature List
Neural networks can be assembled using the following building blocks
 - Fully connected layers
 - Relu Activation
 - Convolutional layers
 - Flatten layer
 - Maxpool
 - Softmax + Crossentropy
 - Xavier initialization for trainable variables
Stochastic Gradient Descent with automatic differentiation is used for training. Inference is a simple forward pass.

## Limitations
 - Only a few types of layers, activations and losses are available
 - Cannot make use of GPUs
 - Cannot be used for implementing Recurrent Neural Networks

## Dependencies
 - numpy
 - matplotlib
 - tqdm

## Steps for replicating the results
 - Rerun ```frontend.ipynb``` notebook to replicate the results
 - See examples in ```frontend.ipynb``` for usage instructions
 - Checkout docstrings in ```backend.py``` for more details
 - data import and plotting utilities are avialable in ```utilities.py```
 - Make sure that the listed dependencies are installed.
