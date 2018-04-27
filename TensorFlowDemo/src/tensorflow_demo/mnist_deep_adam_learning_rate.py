"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dataset.mnist import get_mnist_dataset

from tensorflow_demo.mnist_deep_adam import run_with_specified_args


def main(_):
    # Import data
    mnist = get_mnist_dataset()

    for learning_rate in [0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]:
        print('-' * 80)
        print('\n learning_rate {0} \n'.format(learning_rate))
        run_with_specified_args(mnist, learning_rate=learning_rate, isLogToFile=False)


if __name__ == '__main__':
    tf.app.run(main=main)
