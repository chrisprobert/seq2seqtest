from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils

import sequence_autoencoder


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("size", 16, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("num_examples", 100000, "Number of sequence examples in the model.")
tf.app.flags.DEFINE_integer("max_seq_len", 30, "Maximum sequence length")

FLAGS = tf.app.flags.FLAGS

_buckets = [(5, 5), (10, 10), (15, 15), (25, 25), (50, 50)]
_alphabet_size = 4

def generate_random_autoencoder_examples(numexs) :
    data_set = [[] for _ in _buckets]
    for _ in xrange(numexs) :
        seq_len = np.random.randint(1, FLAGS.max_seq_len)
        seq = np.random.randint(0, _alphabet_size, size=seq_len).tolist()
        for bucket_idx, (insize, outsize) in enumerate(_buckets) :
            if seq_len < insize :
                data_set[bucket_idx].append([seq, seq])
                break
    return data_set

def create_model(session) :
    model = sequence_autoencoder.Seq2SeqAE(
        _alphabet_size, _buckets, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm,
        FLAGS.batch_size, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor)
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
    return model

def create_model_seq2seq(session):
    """Create translation model and initialize or load parameters in session."""
    model = seq2seq_model.Seq2SeqModel(
        _alphabet_size, _alphabet_size, _buckets,
        FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        forward_only=False)
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
    return model

def train():
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)) as sess:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess)
        #model = create_model_seq2seq(sess)

        print('Generating training examples')
        train_set = generate_random_autoencoder_examples(FLAGS.num_examples)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        print('Starting training')
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:

            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                step_time, loss = 0.0, 0.0

                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        train_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()

def main(_):
    train()

if __name__ == "__main__":
  tf.app.run()
