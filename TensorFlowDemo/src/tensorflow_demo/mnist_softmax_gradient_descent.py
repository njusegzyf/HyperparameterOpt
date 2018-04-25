"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners

@see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dataset.mnist import get_mnist_dataset


def main(_):
    # Import data
    mnist = get_mnist_dataset()

    # Parameters
    learning_rate = 0.01
    training_epochs = 10
    batch_size = 100
    display_step = 1

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.random_normal([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    # Initialize the variables (i.e. assign their default value)
    tf.global_variables_initializer().run()

    # Train
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for batch in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            # Fit training using batch data
            c, _ = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y_: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            # print(sess.run(W))
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy,
                   feed_dict={
                       x: mnist.test.images,
                       y_: mnist.test.labels
                   }))


if __name__ == '__main__':
    tf.app.run(main=main)
