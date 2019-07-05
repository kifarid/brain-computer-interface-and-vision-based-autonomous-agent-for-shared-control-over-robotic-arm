from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import OrderedDict
import tensorflow.contrib.slim as slim
from ray.rllib.utils.annotations import override
from ray.rllib.models.model import Model
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.misc import normc_initializer
from ray.rllib.models.preprocessors import get_preprocessor
from gym import spaces

class Q_Network(Model):

    @override(Model)
    def _build_layers_v2(self, inputs, num_output, options={}):
        obs = _unpack_obs(inputs["obs"], self.obs_space)
        input_image = obs['image']
        state = obs['states']
        desired = obs['desired']
        actions = inputs['prev_actions']
        print(input_image.shape ,desired.shape, state.shape)
        state = tf.concat([state, desired, actions], 1)
        net = slim.conv2d(input_image, 64, [1,1], 1, padding='SAME',normalizer_fn=layers.layer_norm, activation_fn=tf.nn.relu, scope='conv1')
        net = slim.max_pool2d(net, [3, 3], scope='pool1')
        net = slim.repeat(net, 6, slim.conv2d, 64, [5, 5], 1, padding='SAME',normalizer_fn=layers.layer_norm, activation_fn=tf.nn.relu, scope='conv2')
        net = slim.max_pool2d(net, [3, 3], scope='pool2')
        F = slim.fully_connected(state, 256,normalizer_fn=layers.layer_norm, activation_fn=tf.nn.relu, scope='fc1')
        F = slim.fully_connected(F, 64,normalizer_fn=layers.layer_norm, activation_fn=tf.nn.relu, scope='fc2')
        net = tf.add(net, tf.reshape(F, [-1, 1, 1, 64]))
        net = slim.repeat(net, 6, slim.conv2d, 64, [3, 3], 1,normalizer_fn=layers.layer_norm, padding='SAME', activation_fn=tf.nn.relu, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], 1,normalizer_fn=layers.layer_norm, padding='SAME', activation_fn=tf.nn.relu, scope='conv4')
        net = slim.fully_connected(tf.contrib.layers.flatten(net), 64,normalizer_fn=layers.layer_norm, activation_fn=tf.nn.relu, scope='fc3')
        net = slim.fully_connected(net, 64, activation_fn=tf.nn.relu,normalizer_fn=layers.layer_norm, scope='fc4')
        output = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, scope='fc_out')
        return output, net

class P_Network(Model):

    @override(Model)
    def _build_layers_v2(self, inputs, num_output, options={}):
        obs = _unpack_obs(inputs["obs"], self.obs_space)
        input_image = obs['image']
        state = obs['states']
        desired = obs['desired']
        state = tf.concat([state, desired],1)
        net = slim.conv2d(input_image, 64, [1,1], 1, padding='SAME',normalizer_fn=layers.layer_norm, activation_fn=tf.nn.relu, scope='conv1')
        net = slim.max_pool2d(net, [3, 3], scope='pool1')
        net = slim.repeat(net, 6, slim.conv2d, 64, [5, 5], 1,normalizer_fn=layers.layer_norm, padding='SAME', activation_fn=tf.nn.relu, scope='conv2')
        net = slim.max_pool2d(net, [3, 3], scope='pool2')
        F = slim.fully_connected(state, 256,normalizer_fn=layers.layer_norm, activation_fn=tf.nn.relu, scope='fc1')
        F = slim.fully_connected(F, 64,normalizer_fn=layers.layer_norm, activation_fn=tf.nn.relu, scope='fc2')
        net = tf.add(net, tf.reshape(F, [-1 ,1, 1, 64]))
        net = slim.repeat(net, 6, slim.conv2d, 64, [3, 3], 1,normalizer_fn=layers.layer_norm, padding='SAME', activation_fn=tf.nn.relu, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], 1,normalizer_fn=layers.layer_norm, padding='SAME', activation_fn=tf.nn.relu, scope='conv4')
        net = slim.fully_connected(tf.contrib.layers.flatten(net), 64, normalizer_fn=layers.layer_norm, activation_fn=tf.nn.relu, scope='fc3')
        net = slim.fully_connected(net, 64, activation_fn=tf.nn.relu,normalizer_fn=layers.layer_norm, scope='fc4')
#added a normalization layer to the pre-last layer
        out = slim.fully_connected(net, 5, activation_fn=tf.nn.sigmoid, scope='fc_out')
        return out, net

def _unpack_obs(obs, space, tensorlib=tf):
    """Unpack a flattened Dict or Tuple observation array/tensor.

    Arguments:
        obs: The flattened observation tensor
        space: The original space prior to flattening
        tensorlib: The library used to unflatten (reshape) the array/tensor
    """

    if (isinstance(space, spaces.Dict)
            or isinstance(space, spaces.Tuple)):
        prep = get_preprocessor(space)(space)
        if len(obs.shape) != 2 or obs.shape[1] != prep.shape[0]:
            raise ValueError(
                "Expected flattened obs shape of [None, {}], got {}".format(
                    prep.shape[0], obs.shape))
        assert len(prep.preprocessors) == len(space.spaces), \
            (len(prep.preprocessors) == len(space.spaces))
        offset = 0
        if isinstance(space, spaces.Tuple):
            u = []
            for p, v in zip(prep.preprocessors, space.spaces):
                obs_slice = obs[:, offset:offset + p.size]
                offset += p.size
                u.append(
                    _unpack_obs(
                        tensorlib.reshape(obs_slice, [-1] + list(p.shape)),
                        v,
                        tensorlib=tensorlib))
        else:
            u = OrderedDict()
            for p, (k, v) in zip(prep.preprocessors, space.spaces.items()):
                obs_slice = obs[:, offset:offset + p.size]
                offset += p.size
                u[k] = _unpack_obs(
                    tensorlib.reshape(obs_slice, [-1] + list(p.shape)),
                    v,
                    tensorlib=tensorlib)
        return u
    else:
        return obs
