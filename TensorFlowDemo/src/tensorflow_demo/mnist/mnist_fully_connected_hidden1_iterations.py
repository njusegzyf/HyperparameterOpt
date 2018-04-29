"""Trains and Evaluates the MNIST network using a feed dictionary.

@see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/fully_connected_feed.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import sys

import tensorflow as tf

from dataset.mnist import get_mnist_dataset
from tensorflow_demo.mnist.mnist_fully_connected import run_training, parse_arguments, FLAGS

FLAGS = None

def main(_):
    # if tf.gfile.Exists(FLAGS.log_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.log_dir)
    # tf.gfile.MakeDirs(FLAGS.log_dir)

    # Get the sets of images and labels for training, validation, and test on MNIST.
    data_sets = get_mnist_dataset()

    result_arr = []
    iterations = 64
    step = 4
    for hidden1_units in range(32, 32 + step * iterations, step):
        print('-' * 100)
        print('Hidden1 units = {0} '.format(hidden1_units))
        FLAGS.hidden1 = hidden1_units
        result_arr.append(run_training(data_sets, FLAGS))

    print('-' * 100)
    for run_result in result_arr:
        print('{0}\t{1}\t{2}\t{3}'.format(run_result.loss_value,
                                          run_result.training_data_eval,
                                          run_result.validation_data_eval,
                                          run_result.test_data_eval))


if __name__ == '__main__':
    FLAGS, unparsed = parse_arguments()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
