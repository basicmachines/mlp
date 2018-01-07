# mlpnet.py
Python implementation of feed-forward multi-layer perceptron (MLP) neural networks using numpy and scipy based on theory taught by Andrew Ng on coursera.org and the Octave code examples from this course.

This is still a work-in-progress.  Still some bugs and fixes to do so don't recommend using this yet...

## Applications

- Classification
- Prediction
- Model Predictive Control (MPC)
- Approximation of non-linear functions


## Main Classes

- `MLPLayer` - Multi-layer perceptron neural network layer class (used by MLPNetwork)
- `MLPNetwork` - Multi-layer perceptron neural network class
- `MLPTrainingData`	- Object to store training data (inputs and outputs).

## Other Functions

- `train` - Used to handle training of a network.
