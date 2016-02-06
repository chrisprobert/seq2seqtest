from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
# from tensorflow.python.training import summary_io

chrom_file = '/Users/chris/Downloads/Homo_sapiens.GRCh38.dna.chromosome.2.fa'
encoding = ['A', 'C', 'G', 'T']
encoding_set = set(encoding)
base2idx = {b: i for i, b in enumerate(encoding)}

_vocab_size = len(encoding)
_GO_ID = _vocab_size
_vocab_size_including_GO = _vocab_size + 1

_seq_length = 7
_batch_size = 256
_reverse_encoder_inputs = True

_lstm_cell_dimension = 8
_lstm_num_layers = 2
_lstm_learn_rate = 0.05
_lstm_max_grad_norm = 1.0

_train_num_iters = 10000
_train_print_loss_every = 100
_train_print_rec_err_every = 100
_train_log_dir = '/tmp/tflogs/'


def check_simple_dna(s):
    for xi in s:
        if xi not in encoding_set:
            return False
    return True


def check_length(s):
    return len(s) >= _seq_length

print('Loading training examples')
num_train_exs = _train_num_iters * _batch_size
with open(chrom_file, 'r') as fp:
    lines = fp.readlines()[1:]
lines = map(str.strip, lines)
lines = map(str.upper, lines)
lines = filter(check_simple_dna, lines)
lines = filter(check_length, lines)
train_exs = []
for _ in xrange(num_train_exs):
    ex = lines[np.random.randint(len(lines))][:_seq_length]
    train_exs.append(np.array(map(base2idx.get, ex)))
print('Loaded {} training examples'.format(len(train_exs)))

# with tf.Session() as sess:
# sess = tf.InteractiveSession()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# tensors to store model state and training data for each batch
seqs = [tf.placeholder(tf.int32, shape=[_seq_length]) for _ in xrange(_batch_size)]
encoder_inputs = [tf.placeholder(tf.int32, shape=[_seq_length]) for _ in xrange(_batch_size)]
decoder_inputs = [tf.placeholder(tf.int32, shape=[_seq_length]) for _ in xrange(_batch_size)]
targets = [tf.placeholder(tf.int32, shape=[_seq_length]) for _ in xrange(_batch_size)]
target_weights = [tf.ones(dtype=tf.float32, shape=[_seq_length]) for _ in xrange(_batch_size)]

# set up the tied seq-to-seq LSTM with given parameters
single_cell = rnn_cell.BasicLSTMCell(_lstm_cell_dimension)
cell = rnn_cell.MultiRNNCell([single_cell] * _lstm_num_layers)
outputs, _ = seq2seq.embedding_tied_rnn_seq2seq(encoder_inputs, decoder_inputs, cell,
                                                _vocab_size_including_GO)
seqloss = seq2seq.sequence_loss_by_example(outputs, encoder_inputs, target_weights,
                                           _vocab_size_including_GO)

tf.train.SummaryWriter(_train_log_dir, sess.graph_def)
global_step = tf.Variable(0, name='global_step', trainable=False)
sess.run(tf.initialize_all_variables())

# Set up the optimizer with gradient clipping
params = tf.trainable_variables()
gradients = tf.gradients(seqloss, params)
optimizer = tf.train.GradientDescentOptimizer(_lstm_learn_rate)
clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                 _lstm_max_grad_norm)
train_op = optimizer.apply_gradients(zip(clipped_gradients, params),
                                     global_step=global_step)

# train_step = tf.train.GradientDescentOptimizer(_lstm_learn_rate).minimize(seqloss)

# Training time!
for itr in xrange(_train_num_iters):
    input_feed = {}
    train_start_idx = itr * _batch_size
    batch = train_exs[train_start_idx:train_start_idx + _batch_size]
    for _i in xrange(_batch_size):
        enc_inputs = batch[_i]
        dec_inputs = np.roll(enc_inputs, 1)
        dec_inputs[0] = _GO_ID
        input_feed[encoder_inputs[_i].name] = enc_inputs
        input_feed[decoder_inputs[_i].name] = dec_inputs

    output_feed = [train_op, seqloss.name]
    for _i in xrange(_batch_size):
        output_feed.append(outputs[_i].name)
    batch_outputs = sess.run(output_feed, input_feed)

    if itr % _train_print_loss_every == 0:
        sloss = np.average(batch_outputs[output_feed.index(seqloss.name)])
        print('{} {}'.format(itr, sloss))
    if itr % _train_print_rec_err_every == 0:
        total_correct = 0
        for seqi, outi in zip(batch, batch_outputs[2:]):
            outi_assign = np.argmax(outi, axis=1)
            total_correct += (outi_assign == seqi).nonzero()[0].shape[0]
        frac_correct = total_correct / (_batch_size * _seq_length)
        print('Fraction correct: {}'.format(frac_correct))
