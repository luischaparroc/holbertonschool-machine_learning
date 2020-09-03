#!/usr/bin/env python3
"""Train"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent
    using early stopping
    """
    def scheduler(epoch):
        """Schedule to learning rate"""
        return alpha / (1 + decay_rate * epoch)

    l_callbacks = list()

    if validation_data:
        if learning_rate_decay:
            l_callbacks.append(K.callbacks.LearningRateScheduler(
                scheduler,
                verbose=1
            ))
        if early_stopping:
            l_callbacks.append(K.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=patience
            ))

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=l_callbacks,
        validation_data=validation_data,
        shuffle=shuffle,
    )
