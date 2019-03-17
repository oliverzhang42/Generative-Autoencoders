import gzip
import os

GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

from scipy import ndimage
from six.moves import urllib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import sys
import scipy.io

print ("PACKAGES LOADED")

def CNN(inputs, _is_training=True):
    x   = tf.reshape(inputs, [-1, 28, 28, 1])
    batch_norm_params = {'is_training': _is_training, 'decay': 0.9, 'updates_collections': None}
    net = slim.conv2d(x, 32, [5, 5], padding='SAME'
                     , activation_fn       = tf.nn.relu
                     , weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
                     , normalizer_fn       = slim.batch_norm
                     , normalizer_params   = batch_norm_params
                     , scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.flatten(net, scope='flatten3')
    net = slim.fully_connected(net, 1024
                    , activation_fn       = tf.nn.relu
                    , weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
                    , normalizer_fn       = slim.batch_norm
                    , normalizer_params   = batch_norm_params
                    , scope='fc4')
    net = slim.dropout(net, keep_prob=0.7, is_training=_is_training, scope='dropout4')  
    out = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return out

# DEFINE MODEL
# PLACEHOLDERS
x  = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10]) #answer
is_training = tf.placeholder(tf.bool, name='MODE')
# CONVOLUTIONAL NEURAL NETWORK MODEL 
y = CNN(x, is_training)
# DEFINE LOSS
with tf.name_scope("LOSS"):
    loss = slim.losses.softmax_cross_entropy(y, y_)
# DEFINE ACCURACY
with tf.name_scope("ACC"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()


# OPEN SESSION
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer(), feed_dict={is_training: False})


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.expand_dims(e_x.sum(axis=1), axis=1)   # only difference

def evaluate(file, fashion=False):
    # RESTORE SAVED NETWORK
    if fashion:
        saver.restore(sess, "./fashion_mnist_weights/model.ckpt")
    else:
        saver.restore(sess, "./mnist_weights/model.ckpt")

    # folders for generated images
    result_folder = './'

    icp = []
    # result_folder = '../GAN/ali_bigan/ali_shell_results/'
    # mat = scipy.io.loadmat(result_folder+ '2_2_2_256_256_1024.mat' )
    # mat = scipy.io.loadmat(result_folder+ '{}.mat'.format(str(k).zfill(3)))
    test_data = np.reshape(np.load(file), (10000, 784))
    if np.max(test_data) > 0.75:
        test_data -= 0.5
    # test_data  = mat['images']
    #  pdb.set_trace()
    # COMPUTE ACCURACY FOR TEST DATA
    batch_size = 100
    test_size   = test_data.shape[0]
    total_batch = int(test_size / batch_size)
    acc_buffer  = []
    preds = []
    for i in range(total_batch):
        offset = (i * batch_size) % (test_size)
        batch_xs = test_data[offset:(offset + batch_size), :]
        y_final = sess.run(y, feed_dict={x: batch_xs, is_training: False})
        pred_softmax = softmax(y_final)
        preds.append(pred_softmax)

    #print(preds[0])

    preds = np.concatenate(preds, 0)
    scores = []
    splits = 10
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      #print("A measurement of the labels of 1 image:")
      #print(np.argmax(part, axis=1))
      #print(np.max(part, axis=1))
      #print(max(part[0].tolist()))
      print("A measurement of Diversity")
      print(np.mean(part, 0))
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))

    icp.append((np.mean(scores) , np.std(scores)))
    print("Inception score is: %.4f, %.4f" % (np.mean(scores) , np.std(scores)))

    return (np.mean(scores), np.std(scores))

    #scipy.io.savemat('ali_inception_50.mat', mdict={'icp': icp})
