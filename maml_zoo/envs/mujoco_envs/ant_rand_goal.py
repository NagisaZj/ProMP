import numpy as np
from meta_policy_search.envs.base import MetaEnv

import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv

class AntRandGoalEnv(MetaEnv, gym.utils.EzPickle, MujocoEnv):
    def __init__(self):
        self.goal_radius = 0.7
        self.cnt = 0
        self.set_task(self.sample_tasks(1)[0])
        MujocoEnv.__init__(self, 'ant.xml', 5)
        gym.utils.EzPickle.__init__(self)

    def sample_tasks(self, num_tasks):
        a = np.random.random(num_tasks) * 1 * np.pi
        r = 1
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.goal_pos = task['goal']

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.goal_pos

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        # mask = (r >= -self.goal_radius).astype(np.float32)
        # r = r * mask
        if r < -self.goal_radius:
            r = -2
        r = r + 2
        return r

    def step(self, action):
        #action = np.clip(action,-0.1,0.1)
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        sparse_reward = self.sparsify_rewards(reward) - ctrl_cost - contact_cost + survive_reward
        # reward = sparse_reward
        # make sparse rewards positive
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        self.cnt = self.cnt + 1
        if self.cnt % 32 == 0:
            self.reset()
            self.cnt = 0
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            sparse_reward=sparse_reward
        )

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def log_diagnostics(self, paths, prefix=''):
        return


if __name__ == "__main__":
    env = AntRandGoalEnv()
    while True:
        env.reset()
        for _ in range(100):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action