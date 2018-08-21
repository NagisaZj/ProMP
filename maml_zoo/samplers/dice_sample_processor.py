from maml_zoo.utils import utils
from maml_zoo.logger import logger
from maml_zoo.samplers.base import SampleProcessor
import numpy as np


class DiceSampleProcessor(SampleProcessor):
    """
    Sample processor for DICE implementations
        - fits a reward baseline (use zero baseline to skip this step)
        - computes adjusted rewards (reward - baseline)
        - normalize adjusted rewards if desired
        - zero-pads paths to max_path_length
        - stacks the padded path data

    Args:
        baseline (Baseline) : a reward baseline object
        max_path_length (int): maximum path length
        discount (float) : reward discount factor
        normalize_adv (bool) : indicates whether to normalize the estimated advantages (zero mean and unit std)
        positive_adv (bool) : indicates whether to shift the (normalized) advantages so that they are all positive
    """

    def __init__(
            self,
            baseline,
            max_path_length,
            discount=0.99,
            normalize_adv=False,
            positive_adv=False,
    ):

        assert 0 <= discount <= 1.0, 'discount factor must be in [0,1]'
        assert max_path_length > 0
        assert hasattr(baseline, 'fit') and hasattr(baseline, 'predict')

        self.max_path_length = max_path_length
        self.baseline = baseline
        self.discount = discount
        self.normalize_adv = normalize_adv
        self.positive_adv = positive_adv

    def process_samples(self, paths, log=False, log_prefix=''):
        """
        Processes sampled paths, This involves:
            - computing discounted rewards
            - fitting a reward baseline
            - computing adjusted rewards (reward - baseline)
            - normalizing adjusted rewards if desired
            - stacking the padded path data
            - creating a mask_array which indicates padded values by zero and original values by one
            - logging statistics of the paths

        Args:
            paths (list): A list of paths of size (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (dict) : Processed sample data. A dict containing the following items with respective shapes:
                    - mask_array: (batch_size, max_path_length)
                    - observations: (batch_size, max_path_length, ndim_act)
                    - actions: (batch_size, max_path_length, ndim_obs)
                    - rewards: (batch_size, max_path_length)
                    - adjusted_rewards: (batch_size, max_path_length)
                    - env_infos: dict of ndarrays of shape (batch_size, max_path_length, ?)
                    - agent_infos: dict of ndarrays of shape (batch_size, max_path_length, ?)

        """
        assert type(paths) == list, 'paths must be a list'
        assert paths[0].keys() >= {'observations', 'actions', 'rewards'}
        assert self.baseline, 'baseline must be specified - use self.build_sample_processor(baseline_obj)'

        # fits baseline, compute advantages and stack path data
        samples_data, paths = self._compute_samples_data(paths)

        # 7) log statistics if desired
        self._log_path_stats(paths, log=log, log_prefix='')

        assert samples_data.keys() >= {'observations', 'actions', 'rewards', 'adjusted_rewards', 'mask_array'}
        return samples_data

    """ helper functions """

    def _compute_samples_data(self, paths):
        assert type(paths) == list

        # 1) compute discounted rewards
        paths = self._compue_discounted_rewards(paths)

        # 2) fit baseline estimator using the path returns and predict the return baselines
        self.baseline.fit(paths)
        all_path_baselines = [self.baseline.predict(path) for path in paths]

        # 3) compute adjusted rewards (r - b)
        paths = self._compute_adjusted_rewards(paths, all_path_baselines)

        # 4) stack path data
        mask_array, observations, actions, rewards, adjusted_rewards, env_infos, agent_infos = self._pad_and_stack_paths(paths)

        # 5) if desired normalize / shift adjusted_rewards
        if self.normalize_adv:
            adjusted_rewards = utils.normalize_advantages(adjusted_rewards)
        if self.positive_adv:
            adjusted_rewards = utils.shift_advantages_to_positive(adjusted_rewards)

        # 6) create samples_data object
        samples_data = dict(
            mask_array=mask_array,
            observations=observations,
            actions=actions,
            rewards=rewards,
            env_infos=env_infos,
            agent_infos=agent_infos,
            adjusted_rewards=adjusted_rewards,
        )

        return samples_data, paths

    def _log_path_stats(self, paths, log=False, log_prefix=''):
        # compute log stats
        average_discounted_return = [sum(path["discounted_rewards"]) for path in paths]
        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        if log == 'reward':
            logger.logkv(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))

        elif log == 'all' or log is True:
            logger.logkv(log_prefix + 'AverageDiscountedReturn', np.mean(average_discounted_return))
            logger.logkv(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))
            logger.logkv(log_prefix + 'NumTrajs', len(paths))
            logger.logkv(log_prefix + 'StdReturn', np.std(undiscounted_returns))
            logger.logkv(log_prefix + 'MaxReturn', np.max(undiscounted_returns))
            logger.logkv(log_prefix + 'MinReturn', np.min(undiscounted_returns))

    def _compue_discounted_rewards(self, paths):
        discount_array = np.cumprod(np.concatenate([np.ones(1), np.ones(self.max_path_length - 1) * self.discount]))

        for path in paths:
            path_length = path['rewards'].shape[0]
            path["discounted_rewards"] = path['rewards'] * discount_array[:path_length]

        return paths

    def _compute_adjusted_rewards(self, paths, all_path_baselines):
        assert len(paths) == len(all_path_baselines)

        for idx, path in enumerate(paths):
            path_baselines = all_path_baselines[idx]
            deltas = path["rewards"] - path_baselines
            path["adjusted_rewards"] = deltas
        return paths

    def _pad_and_stack_paths(self, paths):
        mask_arrays, observations, actions, rewards, adjusted_rewards, env_infos, agent_infos = [], [], [], [], [], [], []
        for path in paths:
            # zero-pad paths if they don't have full length +  create mask_arrays
            path_length = path["observations"].shape[0]
            assert self.max_path_length >= path_length

            mask_arrays.append(self._pad(np.ones(path_length), path_length))
            observations.append(self._pad(path["observations"], path_length))
            actions.append(self._pad(path["actions"], path_length))
            rewards.append(self._pad(path["rewards"], path_length))
            adjusted_rewards.append(self._pad(path["adjusted_rewards"], path_length))
            env_infos.append(dict([(key, self._pad(array, path_length)) for key, array in path["env_infos"].items()]))
            agent_infos.append((dict([(key, self._pad(array, path_length)) for key, array in path["agent_infos"].items()])))

        # stack
        mask_array = np.stack(mask_arrays, axis=0) # shape: (batch_size, max_path_length)
        observations = np.stack(observations, axis=0) # shape: (batch_size, max_path_length, ndim_act)
        actions = np.stack(actions, axis=0) # shape: (batch_size, max_path_length, ndim_obs)
        rewards = np.stack(rewards, axis=0) # shape: (batch_size, max_path_length)
        adjusted_rewards = np.stack(adjusted_rewards, axis=0) # shape: (batch_size, max_path_length)
        env_infos = utils.stack_tensor_dict_list(env_infos) # dict of ndarrays of shape: (batch_size, max_path_length, ?)
        agent_infos = utils.stack_tensor_dict_list(agent_infos) # dict of ndarrays of shape: (batch_size, max_path_length, ?)

        return mask_array, observations, actions, rewards, adjusted_rewards, env_infos, agent_infos

    def _pad(self, array, path_length):
        assert path_length == array.shape[0]
        if array.ndim == 2:
            return np.pad(array, ((0, self.max_path_length - path_length), (0, 0)),  mode='constant')
        elif array.ndim == 1:
            return np.pad(array, (0, self.max_path_length - path_length), mode='constant')
        else:
            raise NotImplementedError