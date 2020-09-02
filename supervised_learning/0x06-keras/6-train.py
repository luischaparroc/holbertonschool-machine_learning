#!/usr/bin/env python3
"""Train"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent
    using early stopping
    """

    if validation_data and early_stopping:
        stopping = [K.callbacks.EarlyStopping(
            patience=patience
        )]
    else:
        stopping = None

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=stopping,
        validation_data=validation_data,
        shuffle=shuffle,
    )
