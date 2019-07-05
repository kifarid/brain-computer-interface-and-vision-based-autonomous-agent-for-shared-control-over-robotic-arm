from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from gym import spaces
from Networks import P_Network
from Networks import Q_Network
from ray.rllib.models.model import Model
from ray.rllib.utils.annotations import override
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.agents.ddpg.ddpg_policy_graph import DDPGPolicyGraph
from ray.rllib.models.misc import normc_initializer

class Q_KOMP(DDPGPolicyGraph):

    @override(DDPGPolicyGraph)
    def _build_q_network(self, inputs, observation_space, action_space, actions):
        XYZlim = 0.8   # limits for states, don't use for goals
        X_max = 1.1    # limit for both goals in x
        Ylim = 0.50    # limit for both goals in y
        Zlim = 1.0     # upper limit for both goals in z
        NZlim = -0.1   # lower limit for both goals in z
        observation_space = spaces.Dict({'achieved': spaces.Box(low=np.tile(np.array([-0.1, -Ylim, NZlim, -0.1, -0.1]),(5,1)),
                                                                       high=np.tile(np.array([X_max, Ylim, Zlim, 1.1, 1.1]),(5,1)), dtype=np.float32),
                                               'desired': spaces.Box(low=np.array([-0.1, -Ylim, NZlim, -0.1, -0.1]),
                                                                       high=np.array([X_max, Ylim, Zlim, 1.1, 1.1]), dtype=np.float32),
                                               'image': spaces.Box(low=0, high=1.1, shape=(234, 234, 4), dtype=np.float32),
                                               'states': spaces.Box(low=np.array([-XYZlim, -XYZlim, 0, -1.1, -0.1]),
                                                                    high=np.array([XYZlim, XYZlim, XYZlim, 1.1, 1.1]), dtype=np.float32),
                                            })
        q_net = Q_Network({'obs':inputs,'prev_actions':actions,}, observation_space, action_space, 1, {"free_log_std":False})
        return q_net.outputs, q_net

    @override(DDPGPolicyGraph)
    def _build_p_network(self, inputs, obs_space, action_space):

        XYZlim = 0.8   # limits for states, don't use for goals
        X_max = 1.1    # limit for both goals in x
        Ylim = 0.50    # limit for both goals in y
        Zlim = 1.0     # upper limit for both goals in z
        NZlim = -0.1   # lower limit for both goals in z
        observation_space = spaces.Dict({'achieved': spaces.Box(low=np.tile(np.array([-0.1, -Ylim, NZlim, -0.1, -0.1]),(5,1)),
                                                                       high=np.tile(np.array([X_max, Ylim, Zlim, 1.1, 1.1]),(5,1)), dtype=np.float32),
                                               'desired': spaces.Box(low=np.array([-0.1, -Ylim, NZlim, -0.1, -0.1]),
                                                                       high=np.array([X_max, Ylim, Zlim, 1.1, 1.1]), dtype=np.float32),
                                               'image': spaces.Box(low=0, high=1.1, shape=(234, 234, 4), dtype=np.float32),
                                               'states': spaces.Box(low=np.array([-XYZlim, -XYZlim, 0, -1.1, -0.1]),
                                                                    high=np.array([XYZlim, XYZlim, XYZlim, 1.1, 1.1]), dtype=np.float32),
                                            })

        p_net = P_Network({'obs':inputs,}, observation_space, action_space, 5, {"free_log_std":False})
        return p_net.outputs, p_net


