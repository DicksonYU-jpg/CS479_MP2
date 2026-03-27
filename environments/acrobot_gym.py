from __future__ import annotations

from typing import Optional, Tuple

from environments.abstract_environment import AbstractEnvironment

import gymnasium as gym
import numpy as np
import imageio


State = Tuple[float, ...]
Action = int
Reward = float


class Acrobot(AbstractEnvironment[State, Action, Reward]):
	def __init__(
		self,
		seed: Optional[int] = None,
	):
		"""
		Wrapper around Gym's Acrobot-v1 environment.

		:param seed: optional RNG seed
		"""
		self._env = gym.make(
			"Acrobot-v1",
			render_mode="rgb_array_list",
		)
		self._seed = seed
		self._end = False
		self._last_reward = 0
		self._state = None
		self.state_dim = self._env.observation_space.shape[0]
		self.n_actions = self._env.action_space.n
		self.reset()

	@property
	def end(self) -> bool:
		return self._end
	
	@property
	def state_space(self):
		obs_space = self._env.observation_space
		lower_bound = obs_space.low
		upper_bound = obs_space.high
		return lower_bound, upper_bound
	
	@property
	def action_space(self):
		action_space = self._env.action_space
		action_list = list(range(action_space.n))
		return action_list
	
	def available(self) -> list[Action]:
		return self.action_space

	def do_action(self, action: Action) -> Tuple[State, Reward]:
		if self._end:
			raise ValueError("The episode has ended, please reset the environment.")

		step_result = self._env.step(action)
		if len(step_result) == 5:
			obs, reward, terminated, truncated, _info = step_result
			self._end = bool(terminated or truncated)
		else:
			obs, reward, done, _info = step_result
			self._end = bool(done)

		
		self._last_reward = float(reward)
		self._state = tuple(float(x) for x in obs)
		return self._state, self._last_reward

	def get_state(self) -> State:
		return self._state

	def reward(self) -> Reward:
		return self._last_reward

	def reset(self):
		reset_result = self._env.reset(seed=self._seed)
		if isinstance(reset_result, tuple):
			obs, _info = reset_result
		else:
			obs = reset_result
		self._end = False
		self._last_reward = 0
		self._state = tuple(float(x) for x in obs)
		return self._state

	def render(self):
		return self._env.render()
	
	def save_gif(self, filename: str):
		
		rgb_array_list = self._env.render()
		images = [np.array(rgb_array) for rgb_array in rgb_array_list]

	
		imageio.mimwrite(filename, images, fps=30)


