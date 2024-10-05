"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict

from typing import Callable, List, Literal, Tuple, Union


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )

    elif name == "pool2d":
        return Pool2D(kernel_shape=kernel_shape, mode=mode, stride=stride, pad=pad)

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]




        W = self.init_weights((self.n_in, self.n_out))
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache: OrderedDict = OrderedDict({"Z": [], "X": []})  # cache for backprop
        self.gradients: OrderedDict = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)})  # parameter gradients initialized to zero
                                           # MUST HAVE THE SAME KEYS AS `self.parameters`

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###
        # perform an affine transformation and activation
        W = self.parameters["W"]
        b = self.parameters["b"]
        
        Z = X @ W + b

        out = self.activation(Z)
        self.cache["Z"] = Z
        self.cache["X"] = X
        
        # store information necessary for backprop in `self.cache`
        return out 
        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  gradient of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        
        # unpack the cache
        W, b, Z, X = self.parameters["W"], self.parameters["b"], self.cache["Z"], self.cache["X"]

        dLdZ = self.activation.backward(Z, dLdY)
        dLdW = X.T @ dLdZ
        dLdX = dLdZ @ W.T
        dLdB = dLdZ.sum(axis = 0, keepdims = True)
        self.gradients["W"], self.gradients["b"] = dLdW, dLdB

        return dLdX


class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({"Z": [], "X": []}) # cache for backprop
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)}) # parameter gradients initialized to zero
                                                                                     # MUST HAVE THE SAME KEYS AS `self.parameters`

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer als supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        #padd it
        paddedX = np.pad(X, [(0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)])

        #set
        stride = self.stride
        rows = int((paddedX.shape[1] - kernel_height) / stride + 1)
        cols = int((paddedX.shape[2] - kernel_width) / stride + 1)
        Z = np.zeros((n_examples, rows, cols, out_channels))

        for d1 in range(rows): 
            for d2 in range(cols): 
                delta_d1 =  d1 * stride + kernel_height
                delta_d2 =  d2 * stride + kernel_width
                X_sliced = paddedX[:, d1*stride : delta_d1, d2*stride : delta_d2 , :]
                Z[:, d1, d2, :] = np.einsum("abcd,bcdx->ax", X_sliced, W) + b
        self.cache["X"] = X
        self.cache["Z"] = Z

        return self.activation(Z)


    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  gradient of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        ### BEGIN YOUR CODE ###
        X, Z = self.cache["X"], self.cache["Z"]
        W = self.parameters["W"]

        height, width, c, d = W.shape
        a, rows, cols, channels = Z.shape
        stride = self.stride
        X_pad = np.pad(X, [(0, 0), (self.pad[0], self.pad[0]),  (self.pad[1], self.pad[1]), (0, 0)])
        # perform a backward pass
        dZ = self.activation.backward(Z, dLdY)
        dX = np.zeros_like(X_pad)
        dW = np.zeros_like(W)

        for d1 in range(rows): 
            for d2 in range(cols): 
                for c in range(channels):
                    #dL/dW = sum sum sum dL/dZ[d1, d2, n] * X[d1 + i, d2+ i, c]
                    
                    dZ_sliced = dZ[:, d1: d1+1, d2: d2+1, c]
                    d1_idx = slice(d1*stride, d1*stride + height)
                    d2_idx = slice(d2*stride, d2*stride + width)

                    dX[:, d1_idx,d2_idx, :] += np.einsum( "abc,bcd->abcd", dZ_sliced, W[ :, :, :, c])
                    dW[:, :, :, c] += np.einsum( "abc,abcd->bcd",dZ_sliced, X_pad[:, d1_idx,d2_idx, :])

                    #dW = dZ_sliced * 

        ### END YOUR CODE ###
        dX = dX[:, self.pad[0] : X.shape[1]+self.pad[0], self.pad[1] : X.shape[2]+self.pad[1], :]

        db = np.einsum("abcx->x", dZ).reshape(1, -1)

        # cache gradients
        self.gradients["W"] = dW
        self.gradients["b"] = db

        return dX

class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        # implement the forward pass
        padded = np.pad(X, pad_width=((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)))

        stride = self.stride
        kernel_height, kernel_width = self.kernel_shape
        batch, x, y, chann = X.shape
        
    
        pooled_rows = int((padded.shape[1] - kernel_height) / stride + 1)
        pooled_cols = int((padded.shape[2] - kernel_width) / stride + 1)

        X_pool = np.zeros((batch, pooled_rows, pooled_cols, chann))

        for row in range(pooled_rows): 
            for col in range(pooled_cols):
                X_pool[:,row , col, :] = self.pool_fn(padded[:, row*stride: row*stride + kernel_height, col*stride:col*stride+kernel_width, :], axis=(1, 2))

        self.cache["X"] = X
        return X_pool

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """
        stride = self.stride
        input_data = self.cache['X']
        batch_size, input_height, input_width, channels = input_data.shape
        filter_height, filter_width = self.kernel_shape

        pad1 = self.pad[0]
        pad2 = self.pad[1]
        output_height = (input_height + 2 * self.pad[0] - filter_height) // stride + 1
        output_width = (input_width + 2 * self.pad[1] - filter_width) // stride + 1

        padded_input = np.pad(input_data, ((0, 0), (pad1,pad1), (pad2,pad2), (0, 0)))
        grad_input = np.zeros_like(padded_input)

        def max_pooling(row, col):
            row_start, row_end = row * stride, row * stride + filter_height
            col_start, col_end = col * stride, col * stride + filter_width


            current_window = padded_input[:, row_start:row_end, col_start:col_end, :]
            window_flat = current_window.reshape(batch_size, -1, channels)

            max_indices = np.argmax(window_flat, axis=1)
            filterClear = np.zeros_like(window_flat)
            filterClear[np.arange(batch_size)[:, None], max_indices, np.arange(channels)] = 1

            window_gradient = filterClear.reshape(current_window.shape) * dLdY[:, row:row+1, col:col+1, :]

            grad_input[:, row_start:row_end, col_start:col_end, :] += window_gradient

        def average_pooling(row, col):
            row_start, row_end = row * stride, row * stride + filter_height

            col_start, col_end = col * stride, col * stride + filter_width

            
            current_window = padded_input[:, row_start:row_end, col_start:col_end, :]
            window_gradient = np.ones_like(current_window) * dLdY[:, row:row+1, col:col+1, :] / (filter_height * filter_width)
            grad_input[:, row_start:row_end, col_start:col_end, :] += window_gradient

        pooling_operations = {
            'max': max_pooling,
            'average': average_pooling
        }

        for row in range(output_height):
            for col in range(output_width):
                pooling_operations[self.mode](row, col)

        return grad_input[:, self.pad[0]:-self.pad[0] or None, self.pad[1]:-self.pad[1] or None, :]

class Flatten(Layer):
    """Flatten the input array."""

    def __init__(self, keep_dim: str = "first") -> None:
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        in_dims = self.cache["in_dims"]
        dX = dLdY.reshape(in_dims)
        return dX
