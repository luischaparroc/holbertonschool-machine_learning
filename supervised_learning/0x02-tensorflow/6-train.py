#!/usr/bin/env python3
"""Train module"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid,
          layer_sizes, activations, alpha, iterations,
          save_path="/tmp/model.ckpt"):
    """Function that builds, trains, and saves a
    neural network classifier
    """

    _, nx = X_train.shape
    _, classes = Y_train.shape

    # Create placeholders
    x, y = create_placeholders(nx, classes)

    # Defining model
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)

    # Adding collections
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # Initialize session
    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            t_c = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            t_a = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            v_c = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            v_a = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            if i == iterations or i % 100 == 0:
                print('After {i} iterations:'.format(i=i))
                print('\tTraining Cost: {cost}'.format(cost=t_c))
                print('\tTraining Accuracy: {accuracy}'.format(accuracy=t_a))
                print('\tValidation Cost: {cost}'.format(cost=v_c))
                print('\tValidation Accuracy: {accuracy}'.format(accuracy=v_a))

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        save_path = saver.save(sess, save_path)
    return save_path
