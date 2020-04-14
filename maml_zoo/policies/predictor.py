from maml_zoo.policies.networks.mlp import create_rnn
from maml_zoo.policies.distributions.diagonal_gaussian import DiagonalGaussian
from maml_zoo.policies.base import Policy
from maml_zoo.utils import Serializable
from maml_zoo.utils.utils import remove_scope_from_name
from maml_zoo.logger import logger

import tensorflow as tf
import numpy as np
from collections import OrderedDict


class RNN(Serializable):

    def __init__(self, input_dim,output_dim,cell_type='lstm'):
        # store the init args for serialization and call the super constructors
        Serializable.quick_init(self, locals())
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.name='predictor'


        self.params = None
        self.input_var = None
        self.hidden_var = None
        self.output_var = None
        self._hidden_state = None
        self.recurrent = True
        self._cell_type = cell_type

        self.build_graph()
        self._zero_hidden = self.cell.zero_state(1, tf.float32)

    def build_graph(self):
        """
        Builds computational graph for policy
        """
        print(self._cell_type)
        with tf.variable_scope(self.name):
            # build the actual policy network
            rnn_outs = create_rnn(name='pred_network',
                                  cell_type=self._cell_type,
                                  output_dim=self.output_dim,
                                  hidden_sizes=[64],
                                  hidden_nonlinearity=tf.tanh,
                                  output_nonlinearity=None,
                                  input_dim=(None, None, self.input_dim,),
                                  )

            self.input_var, self.hidden_var, self.mean_var, self.next_hidden_var, self.cell = rnn_outs

            self.true_rew = tf.placeholder(dtype=tf.float32, shape=(None,128), name='true_rew')
            tr = tf.expand_dims(self.true_rew,1)

            self.regress_loss = tf.norm(tr-self.mean_var,1)



            # symbolically define sampled action and distribution

            # save the policy's trainable variables in dicts
            current_scope = tf.get_default_graph().get_name_scope()
            trainable_policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)
            self.policy_params = OrderedDict([(remove_scope_from_name(var.name, current_scope), var) for var in trainable_policy_vars])

    def get_action(self, observation):
        """
        Runs a single observation through the specified policy and samples an action

        Args:
            observation (ndarray) : single observation - shape: (obs_dim,)

        Returns:
            (ndarray) : single action - shape: (action_dim,)
        """
        observation = np.expand_dims(observation, axis=0)
        action, agent_infos = self.get_actions(observation)
        action, agent_infos = action[0], dict(mean=agent_infos['mean'][0], log_std=agent_infos['log_std'][0])
        return action, agent_infos

    def forward_pass(self, observations):
        """
        Runs each set of observations through each task specific policy

        Args:
            observations (ndarray) : array of observations - shape: (batch_size, obs_dim)

        Returns:
            (ndarray) : array of sampled actions - shape: (batch_size, action_dim)
        """
        observations = np.array(observations)
        assert observations.shape[-1] == self.input_dim
        if observations.ndim == 2:
            observations = np.expand_dims(observations, 1)
        elif observations.ndim == 3:
            pass
        else:
            raise AssertionError
        sess = tf.get_default_session()

        #print(observations.shape)
        #print(self._hidden_state)
        #for t in self._hidden_state:
        #    print(t.shape)
        means, self._hidden_state = sess.run([self.mean_var,  self.next_hidden_var],
                                                     feed_dict={self.input_var: observations,
                                                                self.hidden_var: self._hidden_state})

        assert means.ndim == 3 and means.shape[-1] == self.output_dim

        actions = means


        assert actions.shape == (observations.shape[0], 1, self.output_dim)
        return means [:,0,:]

    def log_diagnostics(self, paths, prefix=''):
        """
        Log extra information per iteration based on the collected paths
        """
        pass




    def distribution_info_keys(self, obs, state_infos):
        """
        Args:
            obs (placeholder) : symbolic variable for observations
            state_infos (dict) : a dictionary of placeholders that contains information about the
            state of the policy at the time it received the observation

        Returns:
            (dict) : a dictionary of tf placeholders for the policy output distribution
        """
        raise ["mean", "log_std"]

    def reset(self, dones=None):
        sess = tf.get_default_session()
        _hidden_state = sess.run(self._zero_hidden)
        #self._hidden_state = sess.run(self.cell.zero_state(len(dones), tf.float32))
        if self._hidden_state is None:
            self._hidden_state = sess.run(self.cell.zero_state(len(dones), tf.float32))
        else:
            #print(type(self._hidden_state))
            if isinstance(self._hidden_state, tf.contrib.rnn.LSTMStateTuple):
                self._hidden_state.c[dones] = _hidden_state.c
                self._hidden_state.h[dones] = _hidden_state.h
            else:
                self._hidden_state[dones] = _hidden_state

    def get_zero_state(self, batch_size):
        sess = tf.get_default_session()
        _hidden_state = sess.run(self._zero_hidden)
        if isinstance(self._hidden_state, tf.contrib.rnn.LSTMStateTuple):
            hidden_c = np.concatenate([_hidden_state.c] * batch_size)
            hidden_h = np.concatenate([_hidden_state.h] * batch_size)
            hidden = tf.contrib.rnn.LSTMStateTuple(hidden_c, hidden_h)
            return hidden
        else:
            return np.concatenate([_hidden_state] * batch_size)
