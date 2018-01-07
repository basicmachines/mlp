#!/usr/bin/env python
"""Neural network simulator

    @author: Bill Tubbs
    Date: 31/12/2017

    This module provides classes to simulate Multi-Layer
    Perceptron (MLP) neural networks for machine learning
    applications.

    Based on theory taught by Andrew Ng on coursera.org
    and the Octave code examples from this course.

    Python modules required to run this module:

    numpy for:
     - multi-dimensional array manipulation
     - arctan, tanh and other useful functions
    scipy for:
     - expit - a fast vectorized version of the Sigmoid function
     - minimize - optimization algorithm used for learning.
    matplotlib.pyplot
     - for plotting surface plots etc.

    TODO list:
    - Confirm if gradients are being calculated correctly
      for activation functions other than sigmoid - may have to
      use an assertion to prevent use of other activation functions
      in cost_function_log().
    - Consider making activation and gradient functions into named
      tuples instead of regular tuples.
    - consider detaching cost_functions (log, mse) from the
      MLPNetwork class
    - otpimize sigmoid derivative calculation (and tanh?) to
      take advantage of fact that derivative is function of
      the sigmoid
    - Try instantiating the A[]'s with the 1.0 values in place
      and then set the remaining values using an assignment
    - find a way to connect the inputs of one network to the
      outputs of another (ideally using a name-object reference
      so no copying is required).
    - Consider whether to move train to a method of network (or
      not).

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


# ---------------- ACTIVATION FUNCTION DEFINITIONS ---------------------

# Each function needs a gradient (derivative) function as well as
# the actuation function itself

# 1. Sigmoid activation function and gradient

# expit is a fast vectorized version of the Sigmoid function, also
# known as the logistic function.  It is imported from the SciPy
# module.
sigmoid = expit

def sigmoid_gradient(z, g=None):
    """sigmoid_gradient(z)

    sigmoid_gradient returns the derivative of the sigmoid function
    (also known as the logistic function) evaluated at z.

    Parameters
    ----------
    z : ndarray
        The ndarray to apply sigmoid_gradient to element-wise.

    g : ndarray
        (optional) An array containing the values sigmoid(z).

    Returns
    -------
    out : ndarray
        An ndarray of the same shape as z. Its entries
        are the derivatives of the corresponding entry of z.
    """

    # TODO: Can we make use of this short-cut if sigmoid(z)
    # has already been calculated?
    if g is None:
        g = sigmoid(z)

    # Compute and return the derivative
    return g*(1.0 - g)


# Came across this alternative way to get the efficiency of
# the Sigmoid derivative
# See here: https://databoys.github.io/Feedforward/

def d_sigmoid(y):
    """derivative of the sigmoid function y = sigmoid(z)
    but as a function of y.

    Returns: y * (1.0 - y)
    """

    return y * (1.0 - y)


# 2. ArcTan activation function and gradient
arctan = np.arctan

def arctan_gradient(z):
    """arctan_gradient(z) returns the derivative of the arctan
    activation function evaluated at z."""

    return 1.0/(z*z + 1.0)


# 3. Hyperbolic tangent (tanh) activation function and gradient
tanh = np.tanh

def tanh_gradient(z):
    """tanh_gradient(z) returns the derivative of the tanh
    activation function evaluated at z."""

    return 1.0 - tanh(z)**2


def d_tanh(y):
    """derivative of the tanh function y = tanh(z)
    but as a function of y.

    Returns: 1 - y*y
    """
    return 1 - y*y


# 4. Linear activation function and gradient

def linear(z):
    """linear(z) is a linear activation function
    that returns z."""

    return z

def linear_gradient(z):
    """linear_gradient(z) returns the derivative of the
    linear activation which is 1.0."""

    return np.ones(z.shape)


# 5. Rectified Linear Unit (ReLU) activation function

def relu(z):
    """relu(z) is the activation function known as ReLU
    (rectified linear unit).
    """
    return z*(z > 0)
    # return np.maximum(0, z)  # Alternative - slightly slower


def relu_gradient(z):
    """relu_gradient(z) returns the gradient of the ReLU
    (Rectified Linear Unit) activation function.
    """

    return (z > 0).astype(float)

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
                     a string.  Current options include 'log' for
                     logistic or 'mse' for mean-squared error.  Note that
                     you must choose a cost function appropriate to the
                     problem you are solving.  For the logistic cost
                     function, all desired outputs must be 0.0 or 1.0.

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
            self.cost_function = self.cost_function_mse
        elif cost_function == 'log':
            self.cost_function = self.cost_function_log
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

    def initialize_weights(self, epsilon=0.12, method='random'):
        """Set the weight values to random numbers in the range -/+ epsilon."""
        if method == 'random':
            self.weights[:] = (
                np.random.rand(self.n_weights)*2.0 - 1
            )*epsilon
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

    def cost_function_log(self, X, Y, weights=None, lambda_param=0.0, jac=True):
        """
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

        # Number of training data points
        m = X.shape[0]

        # Get the weights of each layer as a list of 2-dimensional arrays,
        # either from the network or from the set of weights provided.
        theta = self.get_theta(weights=weights)

        # Prepare list variables for feed-forward computations
        A = [None]*self.n_layers
        Z = [None]*self.n_layers

        # A[0] is the input matrix (network inputs from training data).
        # Create it from a copy of the input data, X with a leading column
        # of ones to simulate the bias terms.
        A[0] = np.concatenate((np.ones((m, 1), dtype=np.float), X), axis=1)

        for j, layer in enumerate(self.layers[1:], start=1):

            # Calculate output values of current layer based on
            # outputs of previous layer
            Z[j] = np.dot(A[j - 1], theta[j].T)

            # Apply the activation function to ouput values
            # Note: only add the column of ones if it is a hidden
            # layer
            #TODO: Wouldn't it be better to instantiate the
            # A[]'s with the 1.0 values in place and then
            # set the remaining values using an assignment?
            if j == self.n_layers - 1:
                A[j] = layer.act_func[0](Z[j])
            else:
                A[j] = np.concatenate(
                    (
                        np.ones((m, 1), dtype=np.float),
                        layer.act_func[0](Z[j])
                    ),
                    axis=1
                )

        # Cost function
        # Negative log likelihood of the Bernoulli distribution
        # (vectorized)
        # This only works with data sets where y = 0.0 or 1.0.
        J = np.sum(-Y*np.log(A[-1]) - (1.0 - Y)*np.log(1.0 - A[-1]))/m
        # %timeit returned 0.448 ms

        # Add regularization terms
        if lambda_param != 0.0:

            for j, layer in enumerate(self.layers[1:], start=1):
                J = J + lambda_param*np.sum(theta[j][:, 1:]**2)/(2.0*m)

        # If jac is set to None or False then don't calculate
        # the gradient
        if not jac:
            return J

        # Otherwise, gradients will be returned in the array grad
        # which has the same dimensions as weights
        grad = np.zeros(self.n_weights, dtype=np.float)

        # sigma, delta and theta_grad arrays for each layer
        # will be stored in the following lists

        # Calcualte dJ/dZ (sigma) for the output layer:
        # TODO: Consider renaming sigma dZ for consistency with
        # Andrew Ng's latest course material
        sigma = [None]*self.n_layers

        # Changes to each weight
        delta = [None]*self.n_layers

        # Partial derivatives of error w.r.t. each weight
        theta_grad = [None]*self.n_layers

        # Now initialise gradient arrays
        first = 0
        previous = self.layers[0]
        for j, layer in enumerate(self.layers[1:], start=1):
            last = first + previous.n_outputs*(layer.n_outputs - 1)
            if last > self.n_weights:
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

        # For negative log-likelihood cost function and with
        # the sigmoid function in the output layer, sigma is
        # simply A - Y:
        sigma[-1] = A[-1] - Y

        # Iterate over the hidden layers to back-propagate
        # the errors
        for j in range(self.n_layers - 2, 0, -1):
            sigma[j] = (
                np.dot(sigma[j + 1], theta[j + 1]) *
                    # TODO: This could be speeded up using the
                    # dsigmoid function or A[-1]*(1 - A[-1]) instead
                    self.layers[j].act_func[1](
                        np.concatenate(
                            (
                                np.ones((m, 1), dtype=np.float),
                                Z[j]
                            ),
                            axis=1
                        )
                    )
            )[:, 1:]

        # Calculate the deltas and gradients for each layer
        for j, layer in enumerate(self.layers[1:], start=1):

            delta[j] = np.dot(sigma[j].T, A[j - 1])

            theta_grad[j][:] = (
                delta[j] + lambda_param * np.concatenate(
                        (
                            np.zeros((theta[j].shape[0], 1), dtype=np.float),
                            theta[j][:, 1:]
                        ),
                        axis=1
                    )
            )/m

        return (J, grad)
        # %timeit returned 5.40 ms

    def cost_function_mse(self, X, Y, weights=None, lambda_param=0.0, jac=True):
        """
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

        # Number of training data points
        m = X.shape[0]

        # Get the weights of each layer as a list of 2-dimensional arrays,
        # either from the network or from the set of weights provided.
        theta = self.get_theta(weights=weights)

        # Prepare list variables for feed-forward computations
        A = [None]*self.n_layers
        Z = [None]*self.n_layers

        # Set A[0] to the input matrix (network inputs from training
        # data) with a column of ones to simulate the bias terms
        A[0] = np.concatenate((np.ones((m, 1), dtype=np.float), X), axis=1)

        for j, layer in enumerate(self.layers[1:], start=1):

            # Calculate output values of current layer based on
            # outputs of previous layer
            Z[j] = np.dot(A[j - 1], theta[j].T)

            # Apply the activation function to ouput values
            # Note: only add the column of ones if it is a hidden
            # layer
            #TODO: Wouldn't it be better to instantiate the
            # A[]'s with the 1.0 values in place and then
            # set the remaining values using an assignment?
            # (A speed test I did suggests it would)
            if j == self.n_layers - 1:
                A[j] = layer.act_func[0](Z[j])
            else:
                A[j] = np.concatenate(
                    (
                        np.ones((m, 1), dtype=np.float),
                        layer.act_func[0](Z[j])
                    ),
                    axis=1
                )

        # Regular mean-squared-error (MSE) cost function
        J = 0.5*np.sum((A[-1] - Y)**2)/m

        # Add regularization terms
        if lambda_param != 0.0:

            for j, layer in enumerate(self.layers[1:], start=1):
                J = J + lambda_param*np.sum(theta[j][:, 1:]**2)/(2.0*m)

        # If jac is set to None or False then don't calculate
        # the gradient
        if not jac:
            return J

        # Otherwise, gradients will be returned in the array grad
        # which has the same dimensions as weights
        grad = np.zeros(self.n_weights, dtype=np.float)

        # sigma, delta and theta_grad arrays for each layer
        # will be stored in the following lists

        # Errors at each node
        sigma = [None]*self.n_layers

        # Changes to each weight
        delta = [None]*self.n_layers

        # Partial derivatives of error w.r.t. each weight
        theta_grad = [None]*self.n_layers

        # Now initialise gradient arrays
        first = 0
        previous = self.layers[0]

        for j, layer in enumerate(self.layers[1:], start=1):
            last = first + previous.n_outputs*(layer.n_outputs - 1)
            if last > self.n_weights:
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

        # Calcualte dJ/dZ (sigma) for the output layer:
        if self.layers[-1].act_func == (sigmoid, sigmoid_gradient):
            sigma[-1] = (A[-1] - Y)*A[-1]*(1 - A[-1])
        elif self.layers[-1].act_func == (tanh, tanh_gradient):
            sigma[-1] = (A[-1] - Y)*(1 - A[-1]**2)
        else:
            sigma[-1] = (A[-1] - Y)*self.layers[-1].act_func[1](Z[-1])

        # Iterate over the hidden layers to back-propagate
        # the errors
        for j in range(self.n_layers - 2, 0, -1):
            sigma[j] = (
                np.dot(sigma[j + 1], theta[j + 1]) *
                    self.layers[j].act_func[1](
                        np.concatenate(
                            (
                                np.ones((m, 1), dtype=np.float),
                                Z[j]
                            ),
                            axis=1
                    )
                )
            )[:, 1:]

        # Calculate the deltas and gradients for each layer
        for j, layer in enumerate(self.layers[1:], start=1):

            #TODO: Confirm - added '*layer.act_func[1](Z[j])' to make
            # this work for MSE. Previously was:
            delta[j] = np.dot(sigma[j].T, A[j - 1])
            #delta[j] = (sigma[j]*layer.act_func[1](Z[j])).T.dot(A[j - 1])

            theta_grad[j][:] = (
                delta[j] + lambda_param * np.concatenate(
                        (
                            np.zeros((theta[j].shape[0], 1), dtype=np.float),
                            theta[j][:, 1:]
                        ),
                        axis=1
                    )
            )/m  # Don't need '*2' here because J above has '0.5*'

        return (J, grad)
        # %timeit returned 5.40 ms

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

        if self.cost_function == self.cost_function_log:
            cost_function = 'log'
        elif self.cost_function == self.cost_function_mse:
            cost_function = 'mse'
        else:
            raise MLPError("Unrecognised cost function assigned to network.")

        s.append("cost_function=%s" % cost_function.__repr__())

        return "MLPNetwork(" + ", ".join(s) + ")"


        self.act_funcs

        return "name=%s, )" % (
            self.name.__repr__(), self.dimensions.__repr__()
        )

    def check_gradients(self, X, y, weights=None, lambda_param=0.0,
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

        # Define a cost function
        def cost_func(p):
            return self.cost_function(
                X, y,
                weights=p,
                jac=True,
                lambda_param=lambda_param
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
        cost = cost_func(weights)

        numgrad = compute_derivative_numerically(cost_func, weights)

        # Visually examine the two gradient computations.  The two
        # columns you get should be very similar.
        for (c1, c2) in zip(numgrad, cost[1]):
            if messages:
                print(c1, c2)
        if messages:
            print('The above two columns you get should be very similar.\n' +
              '(Left-Numerical Gradient, Right-Analytical Gradient)\n\n')

        if (numgrad - cost[1]).sum() == 0.0:
            diff = 0
        else:
            # Evaluate the norm of the difference between two solutions.
            # If you have a correct implementation, and assuming you used
            # EPSILON = 0.0001 in compute_derivative_numerically, then diff
            # below should be less than 1e-9
            diff = np.linalg.norm((numgrad - cost[1]), ord=2) / \
                np.linalg.norm((numgrad + cost[1]), ord=2)

        if messages:
            print("If the backpropagation implementation is correct\n" +
                  "then the relative difference will be small (less\n" +
                  "than 1e-9).\n")
            print("Relative Difference: %g\n" % diff)

        return diff


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


def train(net, data, max_iter=1, update=True, disp=False, method='L-BFGS-B',
          lambda_param=0.0, gtol=1e-6, ftol=0.01):
    """Trains the network (net) on a set of training data (data)
    using the scipy.optimize.minimize function which will minimize
    the network's cost function (net.cost_function) by changing
    the weights (net.weights).

    Returns a scipy.optimize.OptimizeResult object.  See the
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

    cost_func = partial(
        net.cost_function,
        data.inputs,
        data.outputs,
        jac=True,
        lambda_param=lambda_param
    )

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

    print("Solver returned the following message:", str(res.message))

    return res


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
        set_act_funcs(xor, act_funcs)
        print(ndim, act_func_names)

        error = xor.check_gradients(training_data.inputs,
                                 training_data.outputs,
                                 messages=False)
        if np.isnan(error):
            raise ValueError("Calculation error occurred.")

        print("Gradient calc error:", error)

        n_iter = np.random.choice(max_iter_choices)
        train(xor, training_data, max_iter=n_iter)
        cost, grad = xor.cost_function(training_data.inputs, training_data.outputs)
        print("Cost after training:", cost)

        success = True if cost < 0.01 else False
        train_record.append((success, tuple(ndim), tuple(act_func_names)))

    print("\nSummary: %d out of %d tests successful after %d-%d iterations." % \
            (
                sum((item[0] for item in train_record)),
                n,
                max_iter_choices[0],
                max_iter_choices[-1]
            ))

    print("\nFeatures of most successful networks")
    print("Number of layers:")
    results = [(len(item[1]) - 1) for item in train_record if item[0] is True]
    print("\n".join([("%d: %d" % item) for item in top_ranked(results)[0:5]]))

    print("\nTotal number of neurons:")
    results = [sum(item[1][1:]) for item in train_record if item[0] is True]
    print("\n".join([("%d: %d" % item) for item in top_ranked(results)[0:5]]))

    print("\nOutput layer act_func:")
    results = [item[2][-1] for item in train_record if item[0] is True]
    print("\n".join([("%s: %d" % item) for item in top_ranked(results)[0:5]]))

    print("\nAct_func combination:")
    results = [item[2] for item in train_record if item[0] is True]
    print("\n".join([("%s: %d" % item) for item in top_ranked(results)[0:5]]))


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

def compute_derivative_numerically(J, theta):
    """Returns a numerical estimate of the partial derivatives
    of J (the gradients) for each value of theta using
    linear approximation."""

    e = 1.0e-4

    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)

    n = len(theta)

    if n > 1000:
        print("Warning: Computing the numerical gradients with",
               n, "weights could take a long time!")
        input("Program paused. Press enter to continue.")

    for p in range(n):

        # Set perturbation vector
        perturb[p] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)

        if isinstance(loss1, tuple):
            loss1 = loss1[0]

        if isinstance(loss2, tuple):
            loss2 = loss2[0]

        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2.0*e)
        perturb[p] = 0.0

    return numgrad


def compute_function_gradient(f, x):
    """Returns a numerical estimate of the gradient of
    function f at point x."""

    e = 1.0e-4
    return (f(x + e) - f(x - e))/(2.0*e)


# --------------------- START OF MAIN FUNCTION ---------------------


def main():
    """Main function - this will run an example implementation to
    test the module is working."""

    import matplotlib.pyplot as plt

    # Demo - XOR net

    print("\n-------- Demonstration of MLP Network --------")
    print("\nDemo: XOR logic")

    training_data = (
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
        act_funcs=[(sigmoid, sigmoid_gradient), (sigmoid, sigmoid_gradient)],
        cost_function='log'
    )

    # Use this to test tanh or arctan instead
    #xor = MLPNetwork(
    #    ndim,
    #    name="XOR",
    #    act_funcs=[(tanh, tanh_gradient), (tanh, tanh_gradient)],
    #    cost_function='mse'
    #)

    print(xor, "created")

    print("Randomly initialize weights...")
    xor.initialize_weights()

    training_set = MLPTrainingData(ndim=xor.dimensions, data=training_data)

    lambda_param = 0.000001

    (J, grad) = xor.cost_function(
        training_set.inputs,
        training_set.outputs,
        lambda_param=lambda_param
    )

    print("Initial error:", J)

    print("\nCheck gradient functions of activation functions...")

    input("Program paused. Press enter to continue.")

    # Check gradient functions by running checkActFuncGradients
    checkActFuncGradients(activation_functions.values())

    input("Program paused. Press enter to continue.")

    xor.check_gradients(
        training_set.inputs,
        training_set.outputs,
        lambda_param=lambda_param
    )

    input("Program paused. Press enter to continue.")

    print("Begin training...")

    res = train(xor, training_set, max_iter=1000, lambda_param=lambda_param)

    print("Error after learning:", res.fun)

    xor.set_weights(res.x)

    print("Network performance [predictions, training data]:")
    print(np.array_str(
        np.concatenate(
            (
                xor.predict(training_set.inputs),
                training_set.outputs
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
