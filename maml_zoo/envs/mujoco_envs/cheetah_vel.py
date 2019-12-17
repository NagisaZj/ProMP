import numpy as np

from meta_policy_search.envs.base import MetaEnv
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv

class HalfCheetahRandVelEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    def __init__(self):
        self.set_task(self.sample_tasks(1)[0])
        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        gym.utils.EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        return np.random.uniform(0.0, 3.0, (n_tasks, ))

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.goal_velocity = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.goal_velocity

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.5 * 0.1 * np.square(action).sum()
        forward_vel = (xposafter - xposbefore) / self.dt
        reward_run = - np.abs(forward_vel - self.goal_velocity)
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(forward_vel=forward_vel, reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def log_diagnostics(self, paths, prefix=''):
        return

class HalfCheetahRandVelEnvSparse(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    def __init__(self):
        self.goal_radius = 0.5
        self.cnt = 0
        self.set_task(self.sample_tasks(1)[0])
        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        gym.utils.EzPickle.__init__(self)


    def sample_tasks(self, n_tasks):
        return np.random.uniform(0.0, 3.0, (n_tasks, ))

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.goal_velocity = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.goal_velocity

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.5 * 0.1 * np.square(action).sum()
        forward_vel = (xposafter - xposbefore) / self.dt
        reward_run = - np.abs(forward_vel - self.goal_velocity)
        sparse_reward = self.sparsify_rewards(reward_run)
        reward = reward_ctrl + reward_run
        reward = reward_ctrl + sparse_reward
        done = False
        self.cnt = self.cnt + 1
        if self.cnt %64 ==0:
            done = 1
        return ob, reward, done, dict(forward_vel=forward_vel, reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.cnt = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def log_diagnostics(self, paths, prefix=''):
        return

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        #mask = (r >= -self.goal_radius).astype(np.float32)
        #r = r * mask
        if r < - self.goal_radius:
            r = -2
        r = r + 2
        return r