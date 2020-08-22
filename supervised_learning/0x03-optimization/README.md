# Optimization

Project done during **Software Engineering studies** at **Holberton School**. It aims to learn hypermarameters, saddle points, normalizing data, stochastic gradient descent, mini-batch gradient descent, moving average, RMSProp, Adam optimization, learning rate decay and batch normalization.

## Technologies
* Python Scripts are written with Python 3.5
* `Tensorflow`, version 1.12
* `NumPy`, version 1.15

## Files
All of the following files are scripts and programs written in Python:

| Filename | Description |
| -------- | ----------- |
| `0-norm_constants.py` | Function `normalization_constants` that calculates the normalization (standardization) constant of a matrix |
| `1-normalize.py` | Function `normalize` that normalizes (standardizes) a matrix |
| `2-shuffle_data.py` | Function `suffle_data` that shuffles the data points in two matrices the same way |
| `4-moving_average.py` | Function `moving_average` that calculates the weighted moving average of a data set |
| `5-momentum.py` | Function `update_variables_momentum` that updates a variable using the gradient descent with momentum optimization algorithm |
| `6-momentum.py` | Function `create_momentum_op` that creates the training operation for a neural network, using the gradient descent with momentum optimization algorithm |
| `7-RMSProp.py` | Function `update_variables_RMSProp` that updates a variable using the RMSProp optimization algorithm |
| `8-RMSProp.py` | Function `create_RMSProp_op` that creates the training operation for a neural network, using the RMSProp optimization algorithm |
| `9-Adam.py` | Function `update_variables_Adam` that updates a variable in place using the Adam optimization |
