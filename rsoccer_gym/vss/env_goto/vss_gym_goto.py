import math
import random
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv
from rsoccer_gym.Utils import KDTree

class VSSGoToEnv(VSSBaseEnv):
    """This environment controls a single robot in a VSS soccer League 3v3 match 


        Description:
        Observation:
            Type: Box(40)
            Normalized Bounds to [-1.25, 1.25]
            Num             Observation normalized  
            0               Ball X
            1               Ball Y
            2               Ball Vx
            3               Ball Vy
            4 + (7 * i)     id i Blue Robot X
            5 + (7 * i)     id i Blue Robot Y
            6 + (7 * i)     id i Blue Robot sin(theta)
            7 + (7 * i)     id i Blue Robot cos(theta)
            8 + (7 * i)     id i Blue Robot Vx
            9  + (7 * i)    id i Blue Robot Vy
            10 + (7 * i)    id i Blue Robot v_theta
            25 + (5 * i)    id i Yellow Robot X
            26 + (5 * i)    id i Yellow Robot Y
            27 + (5 * i)    id i Yellow Robot Vx
            28 + (5 * i)    id i Yellow Robot Vy
            29 + (5 * i)    id i Yellow Robot v_theta
        Actions:
            Type: Box(2, )
            Num     Action
            0       id 0 Blue Left Wheel Speed  (%)
            1       id 0 Blue Right Wheel Speed (%)
        Reward:
            Sum of Rewards:
                Goal
                Ball Potential Gradient
                Move to Ball
                Energy Penalty
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            5 minutes match time
    """

    def __init__(self, target_margin=0.05, n_targets=1, w_energy=1e-3, w_dist=1):
        super().__init__(field_type=0, n_robots_blue=1, n_robots_yellow=0,
                         time_step=0.025, use_fira=False)

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(2, ), dtype=np.float32)
        n_obs = 7 + 2 * n_targets
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ), dtype=np.float32)

        # Initialize Class Atributes
        self.reward_shaping_total = None
        self.v_wheel_deadzone = 0.05
        self.target_margin = target_margin
        self.n_targets = n_targets
        self.w_energy = w_energy
        self.w_dist = w_dist

        print('Environment initialized')

    def reset(self):
        self.reward_shaping_total = None
        return super().reset()

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    # TODO move render with targets to base 
    def render(self, mode="human") -> None:
        """
        Renders the game depending on
        ball's and players' positions.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if self.view == None:
            from rsoccer_gym.Render import RCGymRender

            self.view = RCGymRender(
                self.n_robots_blue, self.n_robots_yellow, self.field, self.n_targets, simulator="vss"
            )

        return self.view.render_frame(self.frame, self.targets, return_rgb_array=mode == "rgb_array")

    def _frame_to_observations(self):

        observation = []


        observation.append(self.norm_pos(self.frame.robots_blue[0].x))
        observation.append(self.norm_pos(self.frame.robots_blue[0].y))
        observation.append(
            np.sin(np.deg2rad(self.frame.robots_blue[0].theta))
        )
        observation.append(
            np.cos(np.deg2rad(self.frame.robots_blue[0].theta))
        )
        observation.append(self.norm_v(self.frame.robots_blue[0].v_x))
        observation.append(self.norm_v(self.frame.robots_blue[0].v_y))
        observation.append(self.norm_w(self.frame.robots_blue[0].v_theta))

        for t in self.targets:
            observation.append(self.norm_pos(t[0]))
            observation.append(self.norm_pos(t[1]))

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []

        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
        commands.append(Robot(yellow=False, id=0, v_wheel0=v_wheel0,
                              v_wheel1=v_wheel1))

        return commands

    def _calculate_reward_and_done(self):
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {'dist': 0, 'energy': 0}
            for i in range(self.n_targets):
                self.reward_shaping_total[f'target_{i}_score'] = 0
        
        reward = 0
        done = False

        # Check if reached goal
        rbt = [self.frame.robots_blue[0].x, self.frame.robots_blue[0].y]
        p0 = rbt
        p1 = [self.targets[0][0], self.targets[0][1]]
        # If reached next target
        if self.n_reached_targets < self.n_targets and np.linalg.norm(np.array(p1) - np.array(p0)) < self.target_margin:
            # Rotate targets if it was reached
            for i in  range(len(self.targets) - 1):
                self.targets[i] = self.targets[i + 1]
            
            # Calculate target score
            self.reward_shaping_total[f'target_{self.n_reached_targets}_score'] = self.targets_distances[self.n_reached_targets] / (self.steps - self.last_target_steps)
            self.n_reached_targets += 1
            self.last_target_steps = self.steps

        # Calculate distance as sum of distance of each sequential target
        dist = 0
        for t in self.targets:
            p1 = [t[0], t[1]]
            dist += np.linalg.norm(np.array(p1) - np.array(p0))
            p0 = p1

        reward = -dist * self.w_dist
        if dist < self.target_margin:
            # done = True
            reward = 0

        self.reward_shaping_total['dist'] += reward
        energy_pen = self.w_energy * self.__energy_penalty()
        self.reward_shaping_total['energy'] += energy_pen
        reward += energy_pen

        return reward, done

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=self.field.length, y=0.0)

        min_dist = 0.1

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        for i in range(self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        self.targets = []
        self.targets_distances = []
        self.last_target_steps = 0
        self.n_reached_targets = 0
        p0 = [pos_frame.robots_blue[0].x, pos_frame.robots_blue[0].y]
        for i in range(self.n_targets):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())
            places.insert(pos)
            self.targets.append(pos)
            p1 = [pos[0], pos[1]]
            self.targets_distances.append(np.linalg.norm(np.array(p1) - np.array(p0)))
            p0 = p1

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        left_wheel_speed = actions[0] * self.max_v
        right_wheel_speed = actions[1] * self.max_v

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )

        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        # Convert to rad/s
        left_wheel_speed /= self.field.rbt_wheel_radius
        right_wheel_speed /= self.field.rbt_wheel_radius

        return left_wheel_speed , right_wheel_speed

    def __energy_penalty(self):
        '''Calculates the energy penalty'''

        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = - (en_penalty_1 + en_penalty_2)
        return energy_penalty
