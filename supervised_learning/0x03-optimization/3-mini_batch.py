#!/usr/bin/env python3
"""Mini-Batch"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def measure_stats(sess, loss, accuracy, epoch, x, y,
                  X_train, Y_train, X_valid, Y_valid):
    """Measures and prints loss and accuracy of the entire data sets"""
    t_c = sess.run(loss, feed_dict={x: X_train, y: Y_train})
    t_a = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
    v_c = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
    v_a = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
    print('After {epoch} epochs:'.format(epoch=epoch))
    print('\tTraining Cost: {cost}'.format(cost=t_c))
    print('\tTraining Accuracy: {accuracy}'.format(accuracy=t_a))
    print('\tValidation Cost: {cost}'.format(cost=v_c))
    print('\tValidation Accuracy: {accuracy}'.format(accuracy=v_a))


def measure_mini_batch_stats(sess, loss, accuracy, step, x, y,
                             X_train, Y_train):
    """Measures and prints loss and accuracy of epoch data sets"""
    t_c = sess.run(loss, feed_dict={x: X_train, y: Y_train})
    t_a = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
    print('\tStep {step_number}:'.format(step_number=step))
    print('\t\tCost: {step_cost}'.format(step_cost=t_c))
    print('\t\tAccuracy: {step_accuracy}'.format(step_accuracy=t_a))


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains a loaded neural network model using
    mini-batch gradient descent
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(load_path))
        saver.restore(sess, '{}'.format(load_path))

        x, *_ = tf.get_collection('x')
        y, *_ = tf.get_collection('y')
        loss, *_ = tf.get_collection('loss')
        accuracy, *_ = tf.get_collection('accuracy')
        train_op, *_ = tf.get_collection('train_op')

        m, nx = X_train.shape
        steps = [(i, i + batch_size) for i in range(0, m, batch_size)]

        for epoch in range(epochs):
            measure_stats(sess, loss, accuracy, epoch, x, y,
                          X_train, Y_train, X_valid, Y_valid)

            X_s, Y_s = shuffle_data(X_train, Y_train)
            for step, (i, j) in enumerate(steps, start=1):
                sess.run(train_op, feed_dict={x: X_s[i:j], y: Y_s[i:j]})
                if step % 100 == 0:
                    measure_mini_batch_stats(sess, loss, accuracy, step,
                                             x, y, X_s[i:j], Y_s[i:j])

        measure_stats(sess, loss, accuracy, epochs, x, y,
                      X_train, Y_train, X_valid, Y_valid)

        save_path = saver.save(sess, save_path)

    return save_path
