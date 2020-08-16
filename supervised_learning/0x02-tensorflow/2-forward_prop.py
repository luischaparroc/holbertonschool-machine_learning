#!/usr/bin/env python3
"""Forward propagation module"""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Function that creates the forward propagation
    graph for the neural network
    """
    for layer_size, activation in zip(layer_sizes, activations):
        x = create_layer(x, layer_size, activation)
    return x
