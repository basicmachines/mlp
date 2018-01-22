#!/usr/bin/env python
"""Neural Network Simulator

@author: Bill Tubbs
Revision date: 2017-01-07

This Python module provides classes to simulate Multi-Layer
Perceptron (MLP) neural networks for machine learning
applications. It is an efficient vectorized implementation
using numpy arrays and scipy optimization algorithms.

Based on theory taught by Andrew Ng on coursera.org
and adapted from the Octave code examples from this course
as well as updates from the 2017 deeplearning.ai Neural
etworks and Deep Learning course.

Python modules required to run this module:

numpy for:
 - multi-dimensional array manipulation
 - tanh and other useful functions
scipy for:
 - expit - a fast vectorized version of the Sigmoid function
 - minimize - optimization algorithm used for learning
matplotlib.pyplot
 - only needed for the demo in main()
future
 - needed if running Python 2 for builtins such as input()

TODO list:
- Upgrade cost functions so they can be run from network
- Break up common parts of cost_functions into sub-functions
- find a way to connect the inputs of one network to the
  outputs of another (ideally using a name-object reference
  so no copying is required).
- Develop the Trainer class to manage all training
- Consider adding 1.0s to training data inputs to speed up
  training
- Consider whether need to move to separate weights + biases
- Consider making activation and gradient functions into named
  tuples instead of regular tuples - or not.
- Allow for mini-batch updates in Trainer class training
- Change training data subsets into a dictionary for easier
  retrieval
- Remove MLP from class names (module name is sufficient)
- Create a classifier class that inherits from MLPNetwork
- Check lambda parameter is correct - /m /2m etc.
- Add Softmax function
"""

from functools import partial
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from builtins import input


# ------------------------ EXCEPTION CLASS -----------------------------

# See here for more information on user-defined exceptions:
# https://docs.python.org/2/tutorial/errors.html


class MLPError(Exception):
    """Base class for exceptions in this module."""
    pass

# Use these if you want to raise or catch Runtimewarnings in numpy:
np.seterr(all='warn')
#np.seterr(all='raise')

# ---------------- ACTIVATION FUNCTION DEFINITIONS ---------------------

# Each function needs a gradient (derivative) function as well as
# the actuation function itself

# 1. Sigmoid activation function and gradient

# expit is a fast vectorized version of the Sigmoid function, also
# known as the logistic function.  It is imported from the SciPy
# module.
sigmoid = expit

def sigmoid_gradient(z, a=None):
    """sigmoid_gradient(z)

    sigmoid_gradient returns the derivative of the sigmoid function
    (also known as the logistic function) evaluated at z.

    Parameters
    ----------
    z : ndarray
        The ndarray to apply sigmoid_gradient to element-wise.

    a : ndarray
        (optional) An array containing the values sigmoid(z). If
        this has already been calculated, it will speed up the
        computation.

    Returns
    -------
    out : ndarray
        An ndarray of the same shape as z. Its entries
        are the derivatives of the corresponding entry of z.
    """

    if a is None:
        a = sigmoid(z)

    return a*(1.0 - a)


# 2. ArcTan activation function and gradient

# Use numpy vectorized version
arctan = np.arctan

def arctan_gradient(z, a=None):
    """arctan_gradient(z) returns the derivative of the arctan
    activation function evaluated at z.

    Providing a value for a has no effect. The argument is only
    there for consistency with other activation functions."""

    # There is no faster way to compute this using a
    return 1.0/(z**2 + 1.0)


# 3. Hyperbolic tangent (tanh) activation function and gradient

# Use numpy vectorized version
tanh = np.tanh

def tanh_gradient(z, a=None):
    """tanh_gradient(z) returns the derivative of the tanh
    activation function evaluated at z.

    If a=tanh(z) is provided, then the computation will be
    significantly faster."""

    if a is None:
        a = tanh(z)

    return 1.0 - a**2


# 4. Linear activation function and gradient

def linear(z):
    """linear(z) is a linear activation function that
    returns z."""

    return z

def linear_gradient(z, a=None):
    """linear_gradient(z) returns the derivative of the
    linear activation function which is 1.0.

    Providing a value for a has no effect. The argument is only
    there for consistency with other activation functions."""

    return np.ones(z.shape)


# 5. Rectified Linear Unit (ReLU) activation function

def relu(z):
    """relu(z) is the activation function known as ReLU
    (rectified linear unit).
    """

    return z*(z > 0)
    # return np.maximum(0, z)  # Alternative - slightly slower


def relu_gradient(z, a=None):
    """relu_gradient(z) returns the gradient of the ReLU
    (Rectified Linear Unit) activation function at z.

    Providing a value for a has no effect. The argument is only
    there for consistency with other activation functions."""

    return (z > 0).astype(float)

# 6. Softmax

def softmax(z):
    """softmax(z) is a vectorized version of the softmax
    function which can be used to simualate a probability
    distribution (sum of outputs = 0.0)."""

    # This is needed to prevent numerical overflow
    exps = np.exp(z - np.max(z))

    if exps.ndim < 2:
        return exps / np.sum(exps)
    else:
        return exps / exps.sum(axis=1, keepdims=True)

def softmax_gradient(z, y, a):
    """softmax(z) returns the gradients of the softmax
    activation function for the inputs z and desired
    output y.

    Providing a value is optional and reduces computation."""

    # TODO: Not sure this is correct. Need to work on it.
    # Might be better to just implement softmax in the cost_function
    # rather than as an activation function
    # See here: http://cs231n.github.io/neural-networks-case-study/
    grad = np.zeros(z.shape)
    if i == j:
        return a*(1 - z)
    else:
        return -a*z

# This dictionary is used to reference activation functions
# and their derivatives by name

activation_functions = {
    "sigmoid": (sigmoid, sigmoid_gradient),
    "arctan": (arctan, arctan_gradient),
    "tanh": (tanh, tanh_gradient),
    "linear": (linear, linear_gradient),
    "relu": (relu, relu_gradient)
}

# Set the default activation function to use if user does
# not specify one.

default_act_func = activation_functions["sigmoid"]
default_cost_function = 'log'

# Some processor timings

# %timeit sigmoid(z)
# The slowest run took 24.89 times longer than the fastest.
# This could mean that an intermediate result is being cached.
# 1000000 loops, best of 3: 527 ns per loop

# %timeit sigmoid_gradient(z)
# The slowest run took 13.69 times longer than the fastest.
# This could mean that an intermediate result is being cached.
# 100000 loops, best of 3: 2.04 microseconds per loop

# A = sigmoid(z)
# %timeit sigmoid_gradient(z, g=A)
# The slowest run took 14.27 times longer than the fastest.
# This could mean that an intermediate result is being cached.
# 1000000 loops, best of 3: 1.34 microseconds per loop

# %timeit np.arctan(z)
# The slowest run took 21.09 times longer than the fastest.
# This could mean that an intermediate result is being cached.
# 1000000 loops, best of 3: 611 ns per loop

# %timeit arctan_gradient(z)
# The slowest run took 10.43 times longer than the fastest.
# This could mean that an intermediate result is being cached.
# 100000 loops, best of 3: 2.49 microseconds per loop

# %timeit np.tanh(z)
# The slowest run took 24.04 times longer than the fastest.
# This could mean that an intermediate result is being cached.
# 1000000 loops, best of 3: 585 ns per loop

# In [25]: %timeit tanh_gradient(z)
# The slowest run took 12.92 times longer than the fastest.
# This could mean that an intermediate result is being cached.
# 100000 loops, best of 3: 2.18 microseconds per loop


# -------------- MLP NEURAL NETWORK CLASS DEFINITIONS ------------------

# Each Multi-layer perceptron (MLP) network is contained in a MLPNetwork
# instance.  Each MLPNetwork instance contains a list of MLPLayer object
# instances.


class MLPLayer(object):
    """Multi-layer perceptron neural network layer class

    Arguments:
    n_nodes -- the number of nodes in the layer (excluding the
               bias term).

    Keyword arguments:
    input_layer -- a reference to an MLPLayer object that provides
                   the inputs to this layer.  If this layer is the
                   network's input layer then input_layer should be
                   set to None.
    act_func    -- A tuple containing the activation function and
                   its derivative (gradient) function to be used for
                   all neurons in this layer.  If not specified,
                   act_func=default_act_func which should be defined
                   in this module.

    Attributes:
    n_nodes     -- the number of nodes in the layer (excluding the
                   bias term).
    input_layer -- The MLPLayer object that provides the inputs to
                   this layer.  Set to None by default and remains
                   None if this is an input layer.
    n_outputs   -- an integer equal to 1 + the number of outputs from
                   this layer (and equal to the length of the outputs
                   array).
    act_func    -- A tuple containing the activation function and its
                   derivative (gradient) function used for all
                   neurons in this layer.
    outputs     -- one-dimensional numpy array containing the layer's
                   output values in outputs[1:].  These are set to
                   zero initially and outputs[0] is a fixed value
                   always set to 1.0.
    weights     -- Set to None initially, weights is assigned a
                   2-dimensional slice of the network's weights array
                   during the initialisation of a multi-layer
                   network which contains the weights associated with
                   the nodes in this layer (unless this is the input
                   layer, in which case, weights remains None).

    Methods:
    calculate_outputs -- calculates the output values from this layer
                         based on the outputs of the input layer
                         (input_layer).
    """

    def __init__(self, n_nodes, input_layer=None,
                 act_func=default_act_func):

        self.n_nodes = n_nodes
        self.n_outputs = n_nodes + 1
        self.outputs = np.zeros(self.n_outputs, dtype=np.float)
        self.outputs[0] = 1.0
        self.input_layer = input_layer
        self.act_func = act_func
        self.weights = None

    def calculate_outputs(self):
        """Calculate the outputs of each neuron in the layer."""

        if self.input_layer:
            self.outputs[1:] = self.act_func[0](
                np.dot(self.weights, self.input_layer.outputs)
                )
        else:
            raise MLPError("Layer has no inputs.  Note: Cannot "
                           "calculate outputs for layer[0] (the "
                           "input layer).")


class MLPNetwork(object):
    """Multi-layer perceptron neural network class.

    Arguments:
    ndim -- a list of integers that indicate the number of nodes
            in each layer.  Layer 0 is the input layer so ndim[0]
            is also the number of inputs to the network.

    Keyword arguments:
    name          -- (optional) a string to label the network.
    act_funcs     -- Either: (a) a list of tuples containing the
                     activation functions and derivative functions to
                     use for the neurons in each layer (excluding the
                     input layer), or (b) one tuple containing the
                     activation function and its derivative function to
                     be used for all layers. In either case, names of
                     activation functions contained in the dictionary
                     act_funcs may be used (as strings) instead of
                     tuples. If act_funcs is not specified, the default
                     activation function defined in the variable
                     default_act_func will be used for all layers.
    cost_function -- Specify the function to use as a cost function as
                     a string.  Current options include 'log' for the
                     negative log-likelihood or 'mse' for mean-squared
                     error.  Note that you must choose a cost function
                     appropriate to the problem you are solving. For the
                     negative log-likelihood function, desired outputs
                     must be 0.0 or 1.0 and a sigmoid acrtivation function
                     must be used in the output layer.

    Attributes:
    name       -- a string to label the network.
    dimensions -- a list of integers to describe the number of nodes
                  in each layer.  Layer 0 is the input layer.
    n_layers   -- number of layers (including the input layer)
    n_inputs   -- number of inputs.
    n_outputs  -- number of outputs.
    n_nodes    -- number of neurons (number of nodes in layers 1 to
                  n_layers).
    layers     -- list of MLPLayer objects for each layer. Layer 0 is
                  the input layer.
    n_weights  -- number of variable weights.
    weights    -- one-dimensional numpy array of all network weights
                  (weights are set to zero initially).
    gradients  -- one-dimensional numpy array of weight 'gradients'.
                  These are used during learning.
    inputs     -- one-dimensional numpy array of network input values
                  (set to zero initially).
    outputs    -- one-dimensional numpy array of network output values
                  (set to zero initially).  This array may be replaced
                  by any similar-sized array or array slice to allow
                  output values to be written directly to an
                  alternative location without needing to copy values.

    Methods:
    cost_function      -- calculates the cost function for the network and
                          the Jacobian matrix given a set of training data.
                          See the keyword argument 'cost_function' above.
    feed_forward       -- process the network in feed-forward mode. The
                          outputs array will be computed as a result.
    get_theta          -- returns the current weights of each layer as arrays.
    initialize_weights -- initialize the network weights with random numbers
    set_inputs         -- this is a safe method to set the values of the input
                          layer.  You can use 'MLPNetwork.inputs[:] =' but not
                          'MLPNetwork.inputs ='.
    predict            -- makes predictions with the network given a set of
                          inputs.
    set_weights        -- updates the weights with a new set of weight values.
    check_gradients    -- runs a test to compare the calculated gradients with
                          numerical estimates to make sure they are being
                          calculated correctly.
    get_act_funcs      -- Return a list of the activation functions from each
                          layer.
    """

    def __init__(self, ndim, name=None, act_funcs=None,
                 cost_function=default_cost_function):

        self.name = name
        self.dimensions = ndim
        self.n_layers = len(ndim)
        self.n_inputs = ndim[0]
        self.n_outputs = ndim[-1]
        self.n_neurons = sum(self.dimensions[1:])

        if cost_function == 'mse':
            self.cost_function = cost_function_mse
        elif cost_function == 'log':
            self.cost_function = cost_function_log
        else:
            raise MLPError("Cost function choice '%s' not recognised" \
                            % cost_function)

        # If act_funcs or grad_funcs keywords are not provided,
        # use the sigmoid function

        if act_funcs is None:
            act_funcs = default_act_func

        # If a single function is provided, use it for all layers.
        # If not, assume a list was provided.
        # Note, add None as the first item and this will
        # end up assigned to the input layer.

        # Use the dictionary of activation functions to lookup by
        # name (strings)
        if isinstance(act_funcs, str):
            if act_funcs in activation_functions:
                act_funcs = activation_functions[act_funcs]
        if callable(act_funcs[0]) and callable(act_funcs[1]):
            act_funcs = [None] + [act_funcs]*(self.n_layers - 1)
        else:
            assert len(act_funcs) == self.n_layers - 1
            act_funcs = [None] + [activation_functions.get(item,item)
                                  for item in act_funcs]

        # Initialise layers
        self.layers = []
        self.n_weights = 0
        previous = None
        for (d, act_func) in zip(ndim, act_funcs):
            new_layer = MLPLayer(
                d,
                input_layer=previous,
                act_func=act_func
            )
            self.layers.append(new_layer)
            if previous:
                self.n_weights += (new_layer.n_outputs - 1)* \
                                     previous.n_outputs
            previous = new_layer

        # Now initialise weights
        self.weights = np.zeros(self.n_weights, dtype=np.float)
        self.gradients = None

        first = 0
        previous = self.layers[0]
        for j, layer in enumerate(self.layers[1:], start=1):
            last = first + previous.n_outputs*(layer.n_outputs - 1)
            if last > self.n_weights:
                raise MLPError("Error initialising indices of weights arrays.")
            layer.weights = self.weights[first:last]
            try:
                layer.weights.shape = (
                    layer.n_outputs - 1,
                    previous.n_outputs
                )
            except:
                raise MLPError("Error re-shaping the array of weights "
                               " for layer" + str(j))
            previous = layer
            first = last

        # Note that the first output from each layer is always set
        # to 1.0 so the network inputs and outputs arrays must exclude
        # this fixed parameter.  This is easy with numpy ndarrays since
        # slices do not create a copy of the array values.
        self.inputs = self.layers[0].outputs[1:]
        self.outputs = self.layers[self.n_layers - 1].outputs[1:]

    def initialize_weights(self, epsilon=0.01, method='xavier'):
        """Set the network's weights to random values. Currently,
        two methods are implemented:

        Arguments

        epsilon - Defines the variance or range of the random values
                  to use. This is only used if the 'normal' method
                  is selected (see below).

        method  - Method to use:

                  method='normal'. All weights are initialized with
                  random numbers from the same zero-mean normal
                  distribution with a variance of epsilon.

                  method='xavier'. This is the method recommended by
                  Xavier Glorot (year?) which is intended for use with
                  activation functions such as tanh and sigmoid where
                  the gradient of the activation function is 1 at x=0.
                  It uses a zero-mean Gaussian distribution but the
                  variance is set to np.sqrt(1/n[l-1]) where n[l-1] is
                  the number of nodes in the previous (input) layer.

                  method='he'. This is the method recommended by He
                  et al. (2015) which is intended for use with the
                  ReLU activation function. It is the same as the
                  Xavier et al. method but the variance is set to
                  np.sqrt(2/n[l-1]).
        """

        if method == 'normal':
            self.weights[:] = np.random.randn(self.n_weights)*epsilon
        elif method in ('he', 'xavier'):
            n = 2.0 if method == 'he' else 1.0
            for l in range(1, self.n_layers):
                layer = self.layers[l]
                shape = layer.weights.shape
                epsilon = np.sqrt(n/shape[1])
                layer.weights[:] = np.random.standard_normal(shape)*epsilon
        else:
            raise ValueError("Invalid value for keyword argument 'method'")

    def set_inputs(self, inputs):
        """def set_inputs(self, inputs):
            self.inputs[:] = inputs

        Copies the values provided to the network input array.  This is
        provided to avoid mistakenly re-assigning self.inputs to a
        different array by mistake.  This is important because
        self.inputs is a slice into the output array of the input layer.

        In other words, do not do this:
        my_net.inputs = [0.0, 1.0]

        Arguments:
            inputs -- sequence or one-dimensional array of input values."""
        self.inputs[:] = inputs

    def feed_forward(self):
        """Calculate neuron activations in all layers in feed-forward mode."""

        # This might be unnecessary but check that the input
        # attribute is still a view of the output array of the
        # input layer before calculating the network outputs
        if self.inputs.base is not self.layers[0].outputs:
            raise MLPError("Network input attribute has been re-assigned. "
                           "Always use the set_inputs method or only set "
                           "its values.  For example:\n"
                           "my_net.set_inputs([0.0, 1.0])\nor\n"
                           "my_net.inputs[:] = [0.0, 1.0]")

        for layer in self.layers[1:]:
            layer.calculate_outputs()

    def get_theta(self, weights=None):
        """Returns the weights of each layer as a list of
            2-dimensional arrays.  Note: these are not copies
            of the weights so assigning new values is possible.

            If a one-dimensional array of all network weights is
            provided, the list of arrays is created from this
            array instead (not from the current weights in the
            network)."""

        theta = list()

        if weights is None:

            # Return the weight values from the network as a list of
            # arrays
            for layer in self.layers:
                theta.append(layer.weights)

        else:

            if weights.shape != (self.n_weights, ):
                raise ValueError(
                    "Error: weights array provided was not the "
                    "correct shape. Should be " + (self.n_weights, )
                )

            # Create a list of numpy arrays from the 1-dimensional
            # array of weights provided.  First item is empty because
            # there are no weights in input layer
            theta.append(None)

            # Go through each layer and roll up weight values
            # into arrays and then add to the list
            first = 0
            for j, layer in enumerate(self.layers[1:], start=1):
                last = first + layer.weights.size
                if last > self.n_weights:
                    raise MLPError(
                        "Error: too many weights found in network."
                    )
                theta.append(weights[first:last])

                try:
                    theta[-1].shape = layer.weights.shape
                except:
                    raise MLPError(
                        "Error re-shaping the array of weights "
                        "from layer " + str(j)
                    )
                first = last

        return theta

    def predict(self, inputs, weights=None):
        """predict produces a set of predictions using the neural network
        with its current weights or with a new set of weights provided.
        p = predict(inputs) calculates the output predictions for a set
        of inputs.

        Arguments:
            X       -- 2-dimensional array of network inputs
            weights -- (optional) one-dimensional array of weight values."""

        # convert to 2-dimensional array
        inputs = np.asarray(inputs)
        if len(inputs.shape) == 1:
            inputs.shape = (1, inputs.shape[0])

        # Useful values
        m = inputs.shape[0]

        theta = self.get_theta(weights=weights)

        outputs = inputs

        # Calculate the outputs of each layer based on the inputs
        # of the layer below, remembering to add a column of ones
        # to represent the bias terms
        for j, layer in enumerate(self.layers[1:], start=1):

            outputs = layer.act_func[0](
                np.dot(
                    np.concatenate(
                        (np.ones((m, 1), dtype=np.float), outputs),
                        axis=1
                    ),
                    theta[j].T)
            )

        return outputs

    def set_weights(self, weights):
        """Update network weights with set of values provided.

        Arguments:
            weights -- one-dimensional array of weight values."""
        self.weights[:] = weights

    def get_act_funcs(self):
        """Return a list of the activation functions from each
        layer (excluding the input layer)."""

        return [layer.act_func for layer in self.layers[1:]]

    def __repr__(self):

        # Compose a string representation of the object
        s = []

        s.append("ndim=%s" % self.dimensions.__repr__())

        act_funcs = self.get_act_funcs()
        if self.n_neurons > 0:
            if all([x == act_funcs[0] for x in act_funcs]):
                if act_funcs[0] != default_act_func:
                    s.append("act_funcs=%s" % act_funcs[0].__repr__())
            else:
                s.append("act_funcs=%s" % act_funcs.__repr__())

        try:
            if self.name is not None:
                s.append("name=%s" % self.name.__repr__())
        except AttributeError:
            pass

        if self.cost_function is cost_function_log:
            cost_function_name = 'log'
        elif self.cost_function is cost_function_mse:
            cost_function_name = 'mse'
        else:
            raise MLPError("Unrecognised cost function assigned to network.")

        s.append("cost_function=%s" % cost_function_name.__repr__())

        return "MLPNetwork(" + ", ".join(s) + ")"


        self.act_funcs

        return "name=%s, )" % (
            self.name.__repr__(), self.dimensions.__repr__()
        )

    def check_gradients(self, training_data, weights=None, lambda_param=0.0,
                        messages=True):
        """check_gradients uses a numerical approximation to
            check the gradients calculated by the backpropagation
            algorithm.  It outputs the analytically and the
            numerically calculated gradients so you can compare
            them."""

        if messages:
            print('\nChecking backpropagation and gradient calculations...\n')

        # If no weights were provided as an argument, use the network's
        # current weights.
        if weights is None:
            weights = self.weights

        m = training_data.inputs.shape[0]

        arrays = initialize_arrays(self, m)

        # Define a cost function
        def cost_func(p):
            return self.cost_function(
                self,
                training_data,
                weights=p,
                lambda_param=lambda_param,
                jac=True,
                cache=arrays
            )

        # Could use a partial function or lambda function instead
        # cost_func = partial(
        #    self.cost_function,
        #    X, y,
        #    jac=True,
        #    lambda_param=lambda_param
        #)

        # cost_func = lambda p: test_model.cost_function(p, input_layer_size,
        #                  hidden_layer_size, num_labels, X, y, lambda_param)

        # cost_func returns a tuple (cost, grad)
        cost, grad = cost_func(weights)

        numgrad = compute_derivative_numerically(cost_func, weights)

        # Visually examine the two gradient computations.  The two
        # columns you get should be very similar.

        if messages:
            print('Comparison of numerical estimate (Left) with analytical (Right)\n')
            print_list(zip(numgrad, grad))
            print('The above two columns should be very similar.\n')

        diffs = numgrad - grad
        if diffs.sum() == 0.0:
            diff = 0.0
        else:
            # Evaluate the norm of the difference between two solutions.
            # If you have a correct implementation, and assuming you used
            # EPSILON = 0.0001 in compute_derivative_numerically, then diff
            # below should be less than 1e-9
            diff = np.linalg.norm(diffs, ord=2) / \
               (np.linalg.norm(numgrad, ord=2) + np.linalg.norm(grad, ord=2))

        if messages:
            print("If the backpropagation implementation is correct\n" +
                  "then the relative difference will be small (less\n" +
                  "than 1e-7).\n")
            print("Relative Difference: %g\n" % diff)

            biggest = np.argmax(diffs)
            print("Parameter with greatest difference: %s\n" % biggest)
            print(numgrad[biggest], grad[biggest])

        return diff


# ------------------ MLP TRAINING DATA CLASS ----------------------

class MLPTrainingData(object):
    """Data object to store training data (inputs and outputs)
    for use with multi-layer perceptron neural networks
    (e.g. MLPNetwork class).

    Keyword Arguments:
    ndim    -- this should be a list or tuple containing
               integers representing the number of input and
               output values in the training data.  If ndim
               has more than two items, the first item is
               used to set the number of inputs and the last
               is used to set the number of outputs.
    data    -- a two-dimensional array of training
               data.  The first set of columns should contain
               input values and the remaining columns should
               contain the corresponding output values. The
               width of this array must be equal to ndim[0]
               + ndim[-1].
    inputs  -- As an alternative to the above arguments, the
               input data can be specified as a separate
               array.
    outputs -- Similar to inputs above, a separate array of
               output values.  If inputs and outputs are
               specified, do not use data and ndim.
    scaling -- If True, then the input data is normalized
               otherwise None.

    Attributes:
    n_in      -- (int) number of values in input data
    n_out     -- (int) number of values in output data
    data      -- (ndarray) array of training data.  First n_in
                 columns are the input data.  Last n_out
                 columns are the output data.  If training
                 data was initialised using two separate arrays
                 (inputs, outputs) instead of data, then data
                 will be set to None.
    inputs    -- (ndarray) array of input data
    outputs   -- (ndarray) array of output data
    n_subsets -- (int) number of subsets of data.  This attribute
                 will only exist if the method split was called.
    subsets   -- (list) list of data subsets (each subset will be
                 an ndarray).  This attribute will only exist if
                 the method split() was called.
    """

    def __init__(self, data=None, ndim=None, name=None,
                 inputs=None, outputs=None, scaling=None):

        self.name = name

        if data is not None:
            self.n_in = ndim[0]
            self.n_out = ndim[-1]
            self.data = np.asarray(data)

            if len(self.data.shape) != 2:
                raise ValueError("Training data must be a 2-dimensional "
                               "array or nested sequence.")

            if self.data.shape[1] != (self.n_in + self.n_out):
                raise ValueError(
                    "Data provided does not match the number "
                    "of inputs and outputs specified)."
                )

            if np.isnan(self.data).sum(axis=None) > 0:
                raise ValueError(
                    "'Not a number' (NaN) values found in training "
                    "data set provided."
                )

            self.inputs = self.data[:, :self.n_in]
            self.outputs = self.data[:, self.n_in:(self.n_in + self.n_out)]

        else:
            # TODO: This code can be tidied up:
            self.data = None
            self.inputs = np.asarray(inputs, dtype=np.float)
            self.outputs = np.asarray(outputs, dtype=np.float)

            if np.isnan(self.inputs).sum(axis=None) > 0:
                raise ValueError(
                    "'Not a number' (NaN) values found in input "
                    "data set provided."
                )

            if np.isnan(self.outputs).sum(axis=None) > 0:
                raise ValueError(
                    "'Not a number' (NaN) values found in output "
                    "data set provided."
                )

            s = self.inputs.shape
            if len(s) == 2:
                self.n_in = s[1]
            elif len(s) == 1:
                self.n_in = 1
                self.inputs.shape = (s[0], 1)
            else:
                raise ValueError(
                    "Array of input data must be 1 or 2-dimensional."
                    )

            s = self.outputs.shape
            if len(s) == 2:
                self.n_out = s[1]
            elif len(s) == 1:
                self.n_out = 1
                self.outputs.shape = (s[0], 1)
            else:
                raise ValueError(
                    "Array of output data must be 1 or 2-dimensional."
                    )

        if scaling is True:
                # Feature scaling parameters (mean, scale)
                self.mu = np.mean(self.inputs, axis=0)
                self.sigma = np.std(self.inputs, axis=0)

                # Normalise the training data
                self.inputs[:] = (self.inputs - self.mu)*0.25/self.sigma

    def split(self, ratios=(0.75, 0.25), names=('Training set', 'Validation set'), shuffle=True):
        """Split training data points into a number of sub-sets
        (randomly). Useful for separating training data from
        validation and test data.

        Once this function has been executed the training data
        set will have an attribute called subset which is a list
        of ndarrays of the sub-divided data.

        ratios - A list or tuple containing a fraction for each
                 desired subset.
        names  - List or tuple of strings containing names for
                 each sub-set."""

        if sum(ratios) != 1.0:
            raise ValueError("When splitting training data into subsets,"
                             " the sum of the ratios must be 1.")

        n = self.inputs.shape[0]
        n_out = self.outputs.shape[1]

        self.n_subsets = len(ratios)

        num = []
        for r in ratios:
            num.append(int(n*r))
        num[-1] = n - sum(num[:-1])

        # Combine inputs and outputs into one array
        data = np.concatenate(
            (self.outputs, self.inputs),
            axis=1
        )

        # Sort in place, randomly
        if shuffle:
            np.random.shuffle(data)

        self.outputs = data[:, :n_out]
        self.inputs = data[:, n_out:]

        self.subsets = []
        start = 0

        for i, r in enumerate(num):
            finish = start + r
            inputs = self.inputs[start:finish, :]
            outputs = self.outputs[start:finish, :]
            self.subsets.append(MLPTrainingData(inputs=inputs, outputs=outputs, name=names[i]))
            start = finish

    def __repr__(self):

        # Compose a string representation of the object
        s = []

        try:
            if self.data is not None:
                s.append("data=%s" % self.data.__repr__())
        except AttributeError:
            s.append("inputs=%s" % self.inputs.__repr__())
            s.append("outputs=%s" % self.outputs.__repr__())

        try:
            if self.name is not None:
                s.append("name=%s" % self.name.__repr__())
        except AttributeError:
            pass

        return "MLPTrainingData(" + ", ".join(s) + ")"


# ------------------ MLP TRAINER CLASS ----------------------

# Functions and in future a class of object to manage training
# of one or more networks.


def feed_forward(net, A, Z, theta):

    m = A[0].shape[0]

    for j, layer in enumerate(net.layers[1:], start=1):

        # Calculate output values of current layer based on
        # outputs of previous layer
        Z[j][:] = np.dot(A[j - 1], theta[j].T)

        # Apply the activation function to ouput values
        # Note: only add the column of ones if it is a hidden
        # layer
        if j == net.n_layers - 1:
            A[j][:] = layer.act_func[0](Z[j])
        else:
            A[j][:,1:] = layer.act_func[0](Z[j])


def back_prop(net, sigma, A, Z, theta, theta_grad, lambda_param):

    m = A[0].shape[0]

    # Iterate over the hidden layers to back-propagate
    # the errors
    for j in range(net.n_layers - 2, 0, -1):
        sigma[j][:] = (
                np.dot(sigma[j + 1], theta[j + 1][:,1:])*
                net.layers[j].act_func[1](Z[j], A[j][:,1:])
            )

    # Calculate the deltas and gradients for each layer
    for j, layer in enumerate(net.layers[1:], start=1):

        theta_grad[j][:] = np.dot(sigma[j].T, A[j - 1])/m

        # Add component for regularization
        theta_grad[j][:, 1:] += lambda_param*theta[j][:, 1:]/m


def initialize_arrays(net, m):

    # m is number of training data points

    # Set-up matrices needed for vectorized training

    # Prepare list variables for feed-forward computations
    A = [None]*net.n_layers
    Z = [None]*net.n_layers

    # Make each A with a column of ones to simulate the
    # bias terms
    for j, layer in enumerate(net.layers):

        # Prepare matrices for output values:
        if j > 0:
            Z[j] = np.empty((m, layer.n_nodes))

        # Prepare matrices for A:
        if j == net.n_layers - 1:
            A[j] = np.empty((m, layer.n_nodes))
        else:
            A[j] = np.concatenate(
                (
                    np.ones((m, 1), dtype=np.float),
                    np.empty((m, layer.n_nodes))
                ),
                axis=1
            )

    # Prepare array for gradients with the same
    # dimensions as weights
    grad = np.zeros(net.n_weights, dtype=np.float)

    # Partial derivatives of error w.r.t. each weight
    theta_grad = [None]*net.n_layers

    # Changes to each weight
    # TODO: Don't really need this cache
    #delta = [None]*net.n_layers

    # Now initialise gradient and delta arrays
    first = 0
    previous = net.layers[0]

    for j, layer in enumerate(net.layers[1:], start=1):
        last = first + previous.n_outputs*(layer.n_outputs - 1)
        if last > net.n_weights:
            raise MLPError(
                "Error initialising indices of gradients arrays."
            )

        try:
            theta_grad[j] = grad[first:last].reshape((layer.n_outputs \
                            - 1, previous.n_outputs))
        except:
            raise MLPError(
                "Error re-shaping the array of gradients "
                " for layer" + str(j)
            )

        previous = layer
        first = last

    # Errors at each node
    sigma = [None]*net.n_layers

    for j in range(net.n_layers - 1, 0, -1):
        sigma[j] = np.empty((m, net.layers[j].n_nodes))

    return {
        'A': A,
        'Z': Z,
        'sigma': sigma,
        'grad': grad,
        'theta_grad': theta_grad
    }



def train(net, training_data, max_iter=1, update=True, disp=False,
           method='L-BFGS-B', lambda_param=0.0, gtol=1e-6, ftol=0.01):
    """*** TODO: This docstring needs updating ***

    Trains a network (net) on a set of training data (data)
    using the scipy.optimize.minimize function which will minimize
    the a cost function (net.cost_function) by changing the weights
    (net.weights).

    Returns a scipy.optimize.OptimizeResult object. See the
    scipy documentation for a description of its attributes.

    Arguments:
    net          -- MLPNetwork object.
    data         -- MLPTrainingData object.

    Keyword Arguments:
    max_iter     -- Maximum number of iterations of the solver
    update       -- Set to False if you don't want to update the
                    network's weights at the end of the training.
                    Default is True.
    disp         -- Set to True if you want the minimize function
                    to print convergence progress messages during
                    the training.
    method       -- Select the solver to use.  It must be a solver
                    that uses a Jacobian matrix (of gradients).
                    Default is 'L-BFGS-B'.
    lambda_param -- Regularization parameter.  Default is 0.0.
    gtol         -- This is a parameter specific to the 'L-BFGS-B'
                    solver.  The iteration will stop when the
                    maximum gradient is <= gtol.  Default is 1e-6.
    ftol         -- This is a parameter specific to the 'L-BFGS-B'
                    solver.  The iteration will stop when the cost
                    function is <= to ftol.  Default is 0.01.
    """

    X, Y = training_data.inputs, training_data.outputs

    # Number of training data points
    m = X.shape[0]

    # Prepare all arrays (empty)
    # A, Z, sigma, grad, theta_grad
    arrays = initialize_arrays(net, m)

    # Assign training data inputs to A[0]
    arrays['A'][0][:, 1:] = X

    cost_func = partial(
        net.cost_function,
        net,
        training_data,
        lambda_param=lambda_param,
        jac=True,
        cache=arrays
    )

    #cost_func = partial(
    #    net.cost_function,
    #    data.inputs,
    #    data.outputs,
    #    jac=True,
    #    lambda_param=lambda_param
    #)

    res = minimize(
        cost_func,
        net.weights,
        method=method,
        jac=True,
        options={
            'gtol': gtol,
            'ftol': ftol * np.finfo(float).eps,
            'disp': disp,
            'maxiter': max_iter
        }
    )
    # Other options:
    # - CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg

    if update:
        net.weights[:] = res.x

    print("Solver returned the following message:\n%s" % str(res.message))

    return res


def cost_function_log(net, training_data, weights=None,
                      lambda_param=0.0, jac=True, cache=None):
    """
    *** TODO: This docstring needs updating ***

    (J, grad) = cost_function_log(X, Y) computes the cost (J)
    using the logistic cost function* and gradients (grad) of
    the network using back-propagation for the given set of
    training data (X, Y).

    *Note: This cost function is also known as the Bernoulli
    negative log-likelihood and binary cross-entropy.  It
    should only be used for problems such as classification
    where y values are either 0.0 or 1.0.

    Arguments:
    X -- a set of training data points containing m rows of
           network input data
    Y -- a set of desired network outputs for the training
           data containing m rows of output data

    Keyword arguments:
    weights       -- Provide a new set of weights to calculate the cost
                     function (current network weights will not be
                     affected).  If not specified, the cost function will
                     use the current weights stored in the network.
    lambda_param  -- Regularization term.  If not specified then default
                     is lambda_param=0.0 (i.e. no regularization).
    jac           -- If set to None or False then this function does not
                     calculate or return the Jacobian matrix (of gradients).
    """

    # Inputs and desired outputs from training data
    X, Y = training_data.inputs, training_data.outputs

    # Number of examples in training data set
    m = Y.shape[0]

    if cache is None:
        # initialize all arrays as new (empty) arrays
        cache = initialize_arrays(net, m)

        # Assign training data inputs to A[0]
        cache['A'][0][:, 1:] = X

    A = cache['A']
    Z = cache['Z']
    sigma =  cache['sigma']
    grad = cache['grad']
    theta_grad = cache['theta_grad']

    # Get the weights of each layer as a list of 2-dimensional arrays,
    # either from the network or from the set of weights provided.
    theta = net.get_theta(weights=weights)

    # Calculate A and Z
    feed_forward(net, A, Z, theta)

    # Cost function
    # Negative log-likelihood of the Bernoulli distribution
    # (vectorized)
    # This only works with data sets where y = 0.0 or 1.0.
    try:
        J = np.sum(-Y*np.log(A[-1]) - (1.0 - Y)*np.log(1.0 - A[-1]))/m
    except FloatingPointError:
        n_zeros = np.sum(np.any(A[-1] == 0.0))
        n_ones = np.sum(np.any(A[-1] == 1.0))
        messages = ["FloatingPointError occurred."]
        if n_zeros > 0:
            messages.append("%d network output values are 0.0." % n_zeros)
        if n_ones > 0:
            messages.append("%d network output values are 1.0." % n_ones)
        raise ValueError( " ".join(messages))
    # Note: numpy will only raise a warning or error if
    # np.seterr(all='raise')

    # Add regularization terms
    if lambda_param != 0.0:

        for j, layer in enumerate(net.layers[1:], start=1):
            J = J + lambda_param*np.sum(theta[j][:, 1:]**2)/(2.0*m)

    # If jac is set to None or False then don't calculate
    # the gradient and end here
    if not jac:
        return J

    # Otherwise, gradients will be calculated and
    # added to the array grad which has the same
    # dimensions as weights
    # grad = np.zeros(self.n_weights, dtype=np.float)

    # sigma, delta and theta_grad arrays will be calculated
    # for each layer.

    # Calculate dJ/dZ (sigma) for the output layer
    # TODO: Change sigma to dZ for consistency with A Ng course
    # For negative log-likelihood cost function and with
    # the sigmoid function in the output layer, sigma is
    # simply A - Y:
    assert net.layers[-1].act_func is activation_functions["sigmoid"]
    sigma[-1] = A[-1] - Y

    # Back-propagate to calculate derivatives
    back_prop(net, sigma, A, Z, theta, theta_grad, lambda_param)

    return (J, grad)


def cost_function_mse(net, training_data, weights=None,
                      lambda_param=0.0, jac=True, cache=None):
    """
    *** TODO: This docstring needs updating ***

    (J, grad) = cost_function_mse(X, Y) computes the cost (J)
    using the mean-squared-error function* and gradients (grad)
    of the network using back-propagation for the given set of
    training data (X, Y).

    *Note: This cost function is also known as the maximum
    likelihood or sum-squared error method.  It is generally
    useful for regression and function approximation problems.

    Arguments:
    X -- a set of training data points containing m rows of
           network input data
    Y -- a set of desired network outputs for the training
           data containing m rows of output data

    Keyword arguments:
    weights       -- Provide a new set of weights to calculate the cost
                     function (current network weights will not be
                     affected).  If not specified, the cost function will
                     use the current weights stored in the network.
    lambda_param  -- Regularization term.  If not specified then default
                     is lambda_param=0.0 (i.e. no regularization).
    jac           -- If set to None or False then this function does not
                     calculate or return the Jacobian matrix (of gradients).
    """

    # Inputs and desired outputs from training data
    X, Y = training_data.inputs, training_data.outputs

    # Number of examples in training data set
    m = Y.shape[0]

    if cache is None:
        # initialize all arrays as new (empty) arrays
        cache = initialize_arrays(net, m)

        # Assign training data inputs to A[0]
        cache['A'][0][:, 1:] = X

    A = cache['A']
    Z = cache['Z']
    sigma =  cache['sigma']
    grad = cache['grad']
    theta_grad = cache['theta_grad']

    # Get the weights of each layer as a list of 2-dimensional arrays,
    # either from the network or from the set of weights provided.
    theta = net.get_theta(weights=weights)

    # Calculate A and Z
    feed_forward(net, A, Z, theta)

    # Regular mean-squared-error (MSE) cost function
    J = 0.5*np.sum((A[-1] - Y)**2)/m

    # Add regularization terms
    if lambda_param != 0.0:

        for j, layer in enumerate(net.layers[1:], start=1):
            J = J + lambda_param*np.sum(theta[j][:, 1:]**2)/(2.0*m)

    # If jac is set to None or False then don't calculate
    # the gradient and end here
    if not jac:
        return J

    # Otherwise, gradients will be calculated and
    # added to the array grad which has the same
    # dimensions as weights
    # grad = np.zeros(self.n_weights, dtype=np.float)

    # sigma, delta and theta_grad arrays will be calculated
    # for each layer.

    # Calculate dJ/dZ (sigma) for the output layer:
    # TODO: Change sigma to dZ for consistency with A Ng course
    if net.layers[-1].act_func is activation_functions["sigmoid"]:
        sigma[-1][:] = (A[-1] - Y)*A[-1]*(1 - A[-1])
    elif net.layers[-1].act_func is activation_functions["tanh"]:
        sigma[-1][:] = (A[-1] - Y)*(1 - A[-1]**2)
    else:
        sigma[-1][:] = (A[-1] - Y)*net.layers[-1].act_func[1](Z[-1])

    # Back-propagate to calculate derivatives
    back_prop(net, sigma, A, Z, theta, theta_grad, lambda_param)

    return (J, grad)


# THE FOLLOWING FUNCTION IS ONLY FOR TESTING!

def initialize_weights(fan_out, fan_in):
    """initialize_weights Initialize the weights of a layer with fan_in
        incoming connections and fan_out outgoing connections using a fixed
        strategy, this will help you later in debugging
        W = initialize_weights(fan_in, fan_out) initializes the weights
        of a layer with fan_in incoming connections and fan_out outgoing
        connections using a fix set of values

        Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
        the first row of W handles the 'bias' terms."""

    # Set W to zeros
    n = fan_out*(1 + fan_in)

    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    W = np.array(
            np.sin(np.arange(n) + 1)
        ).reshape(
            fan_out,
            1 + fan_in,
            order='F'
            ) / 10.0

    return W


# THE FOLLOWING FUNCTION IS ONLY FOR TESTING!

def random_act_func():
     return np.random.choice(activation_functions.keys())

def set_act_funcs(net, act_funcs):
     for i, act_func in enumerate(act_funcs):
        net.layers[i + 1].act_func = act_func

from collections import defaultdict

def frequency_distribution(sequence):

    freq_dist = defaultdict(int)
    for l in sequence:
        freq_dist[l] += 1

    return freq_dist

def top_ranked(sequence, reverse=True):
    freq_dist = frequency_distribution(sequence)
    return sorted(freq_dist.items(), key=(lambda x: x[1]), reverse=reverse)

def print_list(x):
    print("\n".join([str(i) for i in x]))

def xor_test(n=10, max_iter=100):

    # Training data should create a smooth surface
    data = (
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.5, 0.0, 0.5),
        (0.5, 1.0, 0.5),
        (0.0, 0.5, 0.5),
        (1.0, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )

    training_data = MLPTrainingData(data, ndim=[2, 1])

    print("\n -------------- xor_test -------------- \n")

    print("Test gradient calculations using networks with randomly-")
    print("chosen dimensions and activation functions\n")

    print("Prints ndim, activation funcs, gradient error:")

    max_iter_choices = (int(max_iter/2), max_iter, max_iter*2)
    train_record = []

    for i in range(n):

        ndim = [2]
        for i in range(np.random.randint(1, 3)):
            ndim.append(np.random.randint(2, 10))
        ndim.append(1)

        xor = MLPNetwork(ndim=ndim, cost_function="mse")

        xor.initialize_weights()
        act_func_names = []
        act_funcs = []
        for layer in xor.layers[1:]:
            name = random_act_func()
            act_func_names.append(name)
            act_funcs.append(activation_functions[name])

        # Try negative log-likelihood cost function
        if np.random.random() < 0.25:
            xor.cost_function = cost_function_log
            name = "sigmoid"
            act_func_names[-1] = name
            act_funcs[-1] = activation_functions[name]

        set_act_funcs(xor, act_funcs)

        print("\nDimensions: %s" % str(ndim))
        print("Activation functions: %s" % str(act_func_names))
        print("Cost function: %s" % str(xor.cost_function))

        lambda_param = np.random.choice([0.0, 0.01, 0.1, 0.5, 0.8, 1.0])
        print("lambda: %f" % lambda_param)

        error = xor.check_gradients(training_data,
                                 lambda_param=lambda_param,
                                 messages=False)

        print("Gradient calc error: %.2g" % error)

        # TODO: Training errors around 1.0e-7 occuring with ReLU
        # function - how to avoid?
        if np.isnan(error) or error > 2.0e-6:
            print("\nWARNING: Higher than expected error when checking gradients.")
            print("Re-running gradient check with details...")
            error = xor.check_gradients(training_data,
                                 lambda_param=lambda_param,
                                 messages=True)
            input("\nPress enter to continue.\n")

        n_iter = np.random.choice(max_iter_choices)
        train(xor, training_data, max_iter=n_iter)
        cost, grad = xor.cost_function(xor, training_data)

        print("Cost after training: %7.5f" % cost)

        success = True if cost < 0.01 else False

        train_record.append((success, tuple(ndim), tuple(act_func_names), lambda_param))

    n_successes = sum((item[0] for item in train_record))
    print("\nSummary: %d out of %d tests successful after %d-%d iterations." % \
            (
                n_successes,
                n,
                max_iter_choices[0],
                max_iter_choices[-1]
            ))

    if n_successes > 0:
        print("\nFeatures of most successful networks")
        print("Number of layers:")
        results = [(len(item[1]) - 1) for item in train_record if item[0] is True]
        print_list([("%d: %d" % item) for item in top_ranked(results)[0:5]])
        n_layers_best = top_ranked(results)[0][0]

        print("\nTotal number of neurons:")
        results = [sum(item[1][1:]) for item in train_record if item[0] is True]
        print_list([("%d: %d" % item) for item in top_ranked(results)[0:5]])

        print("\nOutput layer act_func:")
        results = [item[2][-1] for item in train_record if item[0] is True]
        print_list([("%s: %d" % item) for item in top_ranked(results)[0:5]])

        print("\nAct_func combination (%d layers):" % n_layers_best)
        results = [item[2] for item in train_record if item[0] is True and (len(item[1]) - 1) == n_layers_best]
        print_list([("%s: %d" % item) for item in top_ranked(results)[0:5]])

        print("\nLambda:")
        results = [item[3] for item in train_record if item[0] is True]
        print_list([("%s: %d" % item) for item in top_ranked(results)[0:5]])

def check_gradients(lambda_param=0.0):
    """check_gradients Creates a small neural network to check the
        backpropagation gradients
        check_gradients(lambda) Creates a small neural network to check the
        backpropagation gradients, it will output the analytical gradients
        produced by your backprop code and the numerical gradients (computed
        using computeNumericalGradient). These two gradient computations should
        result in very similar values."""

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # Try different activation functions:
    act_funcs = [(sigmoid, sigmoid_gradient), (sigmoid, sigmoid_gradient)]
    #act_funcs = [(arctan, arctan_gradient), (arctan, arctan_gradient)]
    #act_funcs = [(relu, relu_gradient), (sigmoid, sigmoid_gradient)]

    # Initialise the MLP test network for the system model
    ndim = [input_layer_size, hidden_layer_size, num_labels]

    test_model = MLPNetwork(
        ndim,
        name="Test model",
        act_funcs=act_funcs,
        cost_function='mse'
        )

    # We generate some 'random' test data
    theta1 = initialize_weights(hidden_layer_size, input_layer_size)
    theta2 = initialize_weights(num_labels, hidden_layer_size)

    # Reusing initialize_weights to generate X
    X = initialize_weights(m, input_layer_size - 1)
    y = np.zeros((m, num_labels), dtype=np.float)
    for i, v in enumerate(np.arange(1, m+1) % num_labels):
        y[i, v] = 1.0

    # Unroll parameters and store to network
    nn_params = np.concatenate(
        (theta1.ravel(), theta2.ravel()),
        axis=0
        )

    def cost_func(p):
        """Cost function for use with scipy.minimize"""
        # test_model.weights[:] = p
        return test_model.cost_function(
            X, y,
            weights=p,
            lambda_param=lambda_param
        )

    # alternatively could use a lambda function or a partial function
    # cost_func = lambda p: test_model.cost_function(p, input_layer_size,
    #                  hidden_layer_size, num_labels, X, y, lambda_param)

    # cost_func returns a tuple (cost, grad)
    cost = cost_func(nn_params)

    numgrad = compute_derivative_numerically(cost_func, nn_params)

    # Visually examine the two gradient computations.  The two
    # columns you get should be very similar.
    for (c1, c2) in zip(numgrad, cost[1]):
        print(c1, c2)
    print('The above two columns you get should be very similar.\n' + \
          '(Left-Numerical Gradient, Right-Analytical Gradient)\n\n')

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used
    # EPSILON = 0.0001 in compute_derivative_numerically, then diff
    # below should be less than 1e-9
    # TODO: I think this is wrong. Should be extra np.linalg in denominator
    diff = np.linalg.norm((numgrad - cost[1]), ord=2) / \
           np.linalg.norm((numgrad + cost[1]), ord=2)

    print('If your backpropagation implementation is correct, then \n' + \
          'the relative difference will be small (less than 1e-9). \n' + \
          '\nRelative Difference: %g\n' % diff)


def checkActFuncGradients(func_list):
    """Function to check activation gradient functions are
    correct. func_list should be a list of tuples.  Each
    tuple should contain an activation function [item 0]
    and its derivative [item 1]."""

    x_range = np.arange(-1.5, 1.5, 0.5)

    for f in func_list:
        act_func = f[0]
        grad_func = f[1]
        print("\n{}, {}:".format(act_func, grad_func))
        (xa, xn) = \
            grad_func(x_range), \
            compute_function_gradient(act_func, x_range)
        print(" x, grad_func, num. est.")
        for i, x in enumerate(x_range):
            print(" {}, {}, {}".format(x, xa[i], xn[i]))


# THE FOLLOWING FUNCTION IS ONLY FOR TESTING!
# computeNumericalGradient Computes the gradient using "finite differences"
# and gives us a numerical estimate of the gradient.
#   numgrad = computeNumericalGradient(J, theta) computes the numerical
#   gradient of the function J around theta. Calling y = J(theta) should
#   return the function value at theta.

# Notes: The following code implements numerical gradient checking, and
#        returns the numerical gradient.It sets numgrad(i) to (a numerical
#        approximation of) the partial derivative of J with respect to the
#        i-th input argument, evaluated at theta. (i.e., numgrad(i) should
#        be the (approximately) the partial derivative of J with respect
#        to theta(i).)
#

def compute_derivative_numerically(J, theta, epsilon=1.0e-7):
    """Returns a numerical estimate of the partial derivatives
    of J (the gradients) for each value of theta using
    linear approximation."""

    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)

    n = len(theta)

    if n > 1000:
        print("Warning: Computing the numerical gradients with %d" % n)
        print("weights could take a long time!")
        input("Program paused. Press enter to continue.")

    for p in range(n):

        # Set perturbation vector
        perturb[p] = epsilon
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)

        if isinstance(loss1, tuple):
            loss1 = loss1[0]

        if isinstance(loss2, tuple):
            loss2 = loss2[0]

        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2.0*epsilon)
        perturb[p] = 0.0

    return numgrad


def compute_function_gradient(f, x, e=1.0e-7):
    """Returns a numerical estimate of the gradient of
    function f at point x."""

    return (f(x + e) - f(x - e))/(2.0*e)


# --------------------- START OF MAIN FUNCTION ---------------------


def main():
    """Main function - this will run an example implementation to
    test the module is working."""

    import matplotlib.pyplot as plt

    # Demo - XOR net

    print("\n-------- Demonstration of MLP Network --------")
    print("\nDemo: XOR logic")

    data = (
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0)
        #(0.5, 0.0, 0.5),
        #(0.5, 1.0, 0.5),
        #(0.0, 0.5, 0.5),
        #(1.0, 0.5, 0.5)
    )

    ndim = (2, 2, 1)

    xor = MLPNetwork(
        ndim,
        name="XOR",
        act_funcs=["sigmoid", "sigmoid"],
        cost_function='log'
    )

    # Use this to test tanh or arctan instead
    #xor = MLPNetwork(
    #    ndim,
    #    name="XOR",
    #    act_funcs=["tanh", "tanh"],
    #    cost_function='mse'
    #)

    print(str(xor) + "created")

    print("Randomly initialize weights...")
    xor.initialize_weights()

    training_data = MLPTrainingData(ndim=xor.dimensions, data=data)

    lambda_param = 0.01

    #(J, grad) = xor.cost_function(
    #    training_data.inputs,
    #    training_data.outputs,
    #    lambda_param=lambda_param
    #)

    #print("Initial error:", J)

    print("\nCheck gradient functions of activation functions...")

    input("Program paused. Press enter to continue.")

    # Check gradient functions by running checkActFuncGradients
    checkActFuncGradients(activation_functions.values())

    input("Program paused. Press enter to continue.")

    xor.check_gradients(
        training_data,
        lambda_param=lambda_param
    )

    input("Program paused. Press enter to continue.")

    print("Begin training...")

    res = train(xor, training_data, max_iter=1000, lambda_param=lambda_param, disp=True)

    print("Error after learning:", res.fun)

    xor.set_weights(res.x)

    print("Network performance [predictions, training data]:")
    print(np.array_str(
        np.concatenate(
            (
                xor.predict(training_data.inputs),
                training_data.outputs
            ),
            axis=1
        )
        # precision=3,
        # suppress_small=True
    ))

    print("Range of weight values:")
    print(np.min(xor.weights), np.max(xor.weights))

    def z(x, y):
        """Function returns network output value for the given x, y."""
        xor.inputs[:] = (x, y)
        xor.feed_forward()
        return xor.outputs[0]

    # Plot as 3d plot
    from mpl_toolkits.mplot3d import Axes3D

    def show_plot():
        """Show plot"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = y = np.arange(-0.1, 1.1, 0.005)
        X, Y = np.meshgrid(x, y)
        zs = np.array([z(xi, yi) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('x[0]')
        ax.set_ylabel('x[1]')
        ax.set_zlabel('output')
        plt.show()

    print("Generating plot in separate window.")
    print("Close plot window when done..")

    show_plot()


if __name__ == '__main__':
    main()
