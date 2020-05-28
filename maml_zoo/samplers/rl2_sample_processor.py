from maml_zoo.samplers.base import SampleProcessor
import numpy as np
import pickle
from maml_zoo.utils import utils

class RL2SampleProcessor(SampleProcessor):

    def process_samples(self, paths_meta_batch, log=False, log_prefix=''):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths_meta_batch (dict): A list of dict of lists, size: [meta_batch_size] x (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (list of dicts) : Processed sample data among the meta-batch; size: [meta_batch_size] x [7] x (batch_size x max_path_length)
        """
        assert isinstance(paths_meta_batch, dict), 'paths must be a dict'
        assert self.baseline, 'baseline must be specified'

        samples_data_meta_batch = []
        all_paths = []

        for meta_task, paths in paths_meta_batch.items():

            # fits baseline, compute advantages and stack path data
            samples_data, paths = self._compute_samples_data(paths)
            # samples_data['extended_obs'] = np.concatenate([samples_data['observations'], samples_data['actions'],
            #                                                samples_data['rewards'], samples_data['dones']], axis=-1)

            samples_data_meta_batch.append(samples_data)
            all_paths.extend(paths)

        observations, actions, rewards, dones, returns, advantages, env_infos, agent_infos = \
            self._stack_path_data(samples_data_meta_batch)

        overall_avg_reward = np.mean(np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))
        overall_avg_reward_std = np.std(np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))

        for samples_data in samples_data_meta_batch:
            samples_data['adj_avg_rewards'] = (samples_data['rewards'] - overall_avg_reward) / (overall_avg_reward_std + 1e-8)

        # 8) log statistics if desired
        self._log_path_stats(all_paths, log=log, log_prefix=log_prefix)

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
        )
        return samples_data



class ERL2SampleProcessor(SampleProcessor):

    def process_samples(self, paths_meta_batch, log=False, log_prefix='',compute_intr=False,itr=None):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths_meta_batch (dict): A list of dict of lists, size: [meta_batch_size] x (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (list of dicts) : Processed sample data among the meta-batch; size: [meta_batch_size] x [7] x (batch_size x max_path_length)
        """
        assert isinstance(paths_meta_batch, dict), 'paths must be a dict'
        assert self.baseline, 'baseline must be specified'

        samples_data_meta_batch = []
        all_paths = []

        for meta_task, paths in paths_meta_batch.items():

            # fits baseline, compute advantages and stack path data
            if compute_intr:
                samples_data, paths = self._compute_samples_data_intr(paths)
            else:
                samples_data, paths = self._compute_samples_data(paths)
            # samples_data['extended_obs'] = np.concatenate([samples_data['observations'], samples_data['actions'],
            #                                                samples_data['rewards'], samples_data['dones']], axis=-1)

            samples_data_meta_batch.append(samples_data)
            all_paths.extend(paths)

        observations, actions, rewards, dones, returns, advantages, env_infos, agent_infos, ori_rews = \
            self._stack_path_data(samples_data_meta_batch)

        overall_avg_reward = np.mean(np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))
        overall_avg_reward_std = np.std(np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))

        for samples_data in samples_data_meta_batch:
            samples_data['adj_avg_rewards'] = (samples_data['rewards'] - overall_avg_reward) / (overall_avg_reward_std + 1e-8)

        # 8) log statistics if desired
        self._log_path_stats(all_paths, log=log, log_prefix=log_prefix)

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
            ori_rewards=ori_rews
        )
        if 'test' in log_prefix:
            filename='/home/zj/Desktop/ProMP/info.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(samples_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return samples_data

    def _stack_path_data(self, paths):
        max_path = max([len(path['observations']) for path in paths])

        observations = self._stack_padding(paths, 'observations', max_path)
        actions = self._stack_padding(paths, 'actions', max_path)
        rewards = self._stack_padding(paths, 'rewards', max_path)
        dones = self._stack_padding(paths, 'dones', max_path)
        returns = self._stack_padding(paths, 'returns', max_path)
        advantages = self._stack_padding(paths, 'advantages', max_path)
        ori_rews = self._stack_padding(paths, 'advantages', max_path)
        env_infos = utils.stack_tensor_dict_list([path["env_infos"] for path in paths], max_path)
        agent_infos = utils.stack_tensor_dict_list([path["agent_infos"] for path in paths], max_path)

        return observations, actions, rewards, dones, returns, advantages, env_infos, agent_infos, ori_rews

    def _compute_samples_data_intr(self, paths):
        assert type(paths) == list

        # 1) compute discounted rewards (returns)

        for idx, path in enumerate(paths):
            path['ori_rewards'] = path['rewards']
            self.predictor.reset(dones=[True] * 128)
            predicted_rews = self.predictor.forward_pass(path['observations'])[:,0]
            #print(predicted_rews.shape,path['rewards'].shape)
            path["rewards"] = abs(predicted_rews-path['rewards'])*self.intr_rew_weight+path['rewards']
            path["returns"] = utils.discount_cumsum(path["rewards"], self.discount)

        # 2) fit baseline estimator using the path returns and predict the return baselines
        self.baseline.fit(paths, target_key="returns")
        all_path_baselines = [self.baseline.predict(path) for path in paths]

        # 3) compute advantages and adjusted rewards
        paths = self._compute_advantages(paths, all_path_baselines)

        # 4) stack path data
        observations, actions, rewards, dones, returns, advantages, env_infos, agent_infos ,ori_rewards= self._concatenate_path_data(paths)

        # 5) if desired normalize / shift advantages
        if self.normalize_adv:
            advantages = utils.normalize_advantages(advantages)
        if self.positive_adv:
            advantages = utils.shift_advantages_to_positive(advantages)

        # 6) create samples_data object
        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
            ori_rewards=ori_rewards
        )

        return samples_data, paths

    def _concatenate_path_data(self, paths):
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        dones = np.concatenate([path["dones"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        env_infos = utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos = utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])
        os = np.concatenate([path["ori_rewards"] for path in paths])
        return observations, actions, rewards, dones, returns, advantages, env_infos, agent_infos, os