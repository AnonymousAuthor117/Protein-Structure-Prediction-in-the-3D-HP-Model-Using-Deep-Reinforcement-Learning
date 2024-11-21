import sys
from collections import OrderedDict
import gym
from gym import (spaces, utils, logger)
import numpy as np

from hpsandbox_util import (
    plot_HPSandbox_conf,
    move_LFR_direction,
)

#action space
ACTION_TO_STR = {
    0: 'L',
    1: 'F',
    2: 'R',
    3: 'U',
    4: 'D',
    }
class ThreeActionStateEnv(gym.Env):
    def __init__(self, seq, obs_output_mode = "tuple"):
        self.seq = seq
        self.obs_output_mode = obs_output_mode

        #initializing the values
        self.reset()
        if len(self.seq) <= 2:
            return

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=5, shape=(len(self.seq)-2,), dtype=int)
        self.first_turn_left = False

        #logging
        print("ThreeActionStateEnv init with attributes:")
        print("self.seq = ", self.seq)
        print("len(self.seq) = ", len(self.seq))
        print("self.obs_output_mode = ", self.obs_output_mode)

        print("self.state = ", self.state)
        print("self.actions = ", self.actions)

        print("self.action_space:")
        print(self.action_space)
        print("self.observation_space:")
        print(self.observation_space)
        print("self.observation_space.high, low:")
        print(self.observation_space.high)
        print(self.observation_space.low)
        print("self.observation_space.shape:")
        print(self.observation_space.shape)
        print("self.observation_space.dtype, self.action_space.dtype")
        print(self.observation_space.dtype, self.action_space.dtype)

        print("self.first_turn_left = ", self.first_turn_left)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("%r (%s) invalid" % (action, type(action)))

        if (action != 1) and (self.first_turn_left is False):
            if action == 2:
                action = 0
            self.first_turn_left = True

        self.last_action = action
        is_trapped = False

        p2 = list(self.state.keys())[-1]
        p1 = list(self.state.keys())[-2]

        next_move = move_LFR_direction(
            p1=p1,
            p2=p2,
            move_direction=action,
        )

        idx = len(self.state)
        if next_move in self.state:
            #return self.observe(), 0, False, {"is_trapped": True}  # Adjusted to return valid state
            return (None, None, False, {})
        else:
            self.actions.append(action)
            try:
                self.state.update({next_move : self.seq[idx]})
            except IndexError:
                logger.error('All molecules have been placed! Nothing can be added to the protein chain.')
                raise
            if len(self.state) < len(self.seq):
                if set(self._get_adjacent_coords(next_move).values()).issubset(self.state.keys()):
                    is_trapped = True

        obs = self.observe()
        self.done = True if (len(self.state) == len(self.seq) or is_trapped) else False
        reward = self._compute_reward()
        info = {
            'chain_length' : len(self.state),
            'seq_length'   : len(self.seq),
            'actions'      : [ACTION_TO_STR[i] for i in self.actions],
            'is_trapped'   : is_trapped,
            'state_chain'  : self.state,
            "first_turn_left": self.first_turn_left,
        }
        return (obs, reward, self.done, info)
    
    def observe(self):
        action_chain = self.actions
        native_obs = np.zeros(shape=(len(self.seq)-2,), dtype=int)

        for i, item in enumerate(action_chain):
            native_obs[i] = item+1

        quaternary_tuple = native_obs
        if self.obs_output_mode == "tuple":
            return quaternary_tuple

    def reset(self, seed=None, options=None):
        self.actions = []
        self.last_action = None
        self.prev_reward = 0

        self.state = OrderedDict(
            {
                (0, 0, 0): self.seq[0],
                (0, 0, -1): self.seq[1],
            }
        )
        self.done = len(self.seq) == 2
        obs = self.observe()
        self.first_turn_left = False
        return obs

    def render(self, mode="draw", pause_t=0.0, save_fig=False, save_path="", score=2022, optima_idx=0):
        plot_HPSandbox_conf(
                list(self.state.items()),
                mode=mode,
                pause_t=pause_t,
                save_fig=save_fig,
                save_path=save_path,
                score=score,
                optima_idx=optima_idx,
                info={
                    'chain_length' : len(self.state),
                    'seq_length'   : len(self.seq),
                    'actions'      : [ACTION_TO_STR[i] for i in self.actions],
                },
            )
     def _get_adjacent_coords(self, coords):
        x, y, z = coords
        adjacent_coords = {
            0 : (x - 1, y, z),
            1 : (x, y - 1, z),
            2 : (x, y + 1, z),
            3 : (x + 1, y, z),
            4 : (x, y, z + 1),
            5 : (x, y, z - 1),
        }
        return adjacent_coords

    def _compute_reward(self):
        curr_reward = self._compute_free_energy(self.state)
        state_E = curr_reward
        step_E = curr_reward - self.prev_reward
        self.prev_reward = curr_reward
        reward = curr_reward if self.done else 0
        return (-state_E, -step_E, -reward)

    def is_adjacent(self, coord1, coord2):
        x1, y1, z1 = coord1
        x2, y2, z2 = coord2

        if (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) == 1:
            return True
        else:
            return False

    def _compute_free_energy(self, chain):
        energy = 0
        coordinates = list(chain.keys())
        sequence = list(chain.values())

        for i in range(len(coordinates)):
            if sequence[i] == 'H':
                for j in range(i + 1, len(coordinates)):
                    if sequence[j] == 'H':
                        # Ensure the H-H bond is not part of the backbone
                        if self.is_adjacent(coordinates[i], coordinates[j]) and abs(i - j) > 1:
                            energy -= 1

        return energy

    def seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        self.action_space.seed(seed)
        np.random.seed(seed)
        return [seed]
