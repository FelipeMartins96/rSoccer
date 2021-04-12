import math
import random
from typing import Dict

import gym
import numpy as np
from rc_gym.Entities import Ball, Frame, Robot
from rc_gym.ssl.ssl_gym_base import SSLBaseEnv


class SSLPassEnduranceEnv(SSLBaseEnv):
    """The SSL robot needs to make a goal with contested possession


        Description:

        Observation:
            Type: Box(4 + 5*n_robots_blue)
            Normalized Bounds to [-1.2, 1.2]
            Num      Observation normalized  
            0->3     Ball [X, Y, V_x, V_y]
            4->7    id 0 Blue [X, Y, sin(theta), cos(theta)]
            8->11    id 1 Blue [X, Y]


        Actions:
            Type: Box(3,)
            Num     Action
            0       id i Blue Angular Speed  (%)
            1       id i Blue Kick x Speed  (%)
            2       id i Blue Dribbler  (%) (true if % is positive)

        Reward:

        Starting State:

        Episode Termination:
            30 seconds (1200 steps) or wrong pass
    """
    original_vec = None
    max_dist = 100
    actions = {}
    shooted = False

    def __init__(self):
        super().__init__(field_type=2, n_robots_blue=2,
                         n_robots_yellow=0, time_step=0.025)
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(3, ),
                                           dtype=np.float32)

        n_obs = 4 + 4 + 2
        self.holding_steps = 0
        self.stopped_steps = 0
        self.recv_angle = 270
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs,),
                                                dtype=np.float32)
        self.receiver_id = 1

        print('Environment initialized')

    def _frame_to_observations(self):
        observation = list()

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))
        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            if i == 0:
                observation.append(
                    np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
                )
                observation.append(
                    np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
                )
        return np.array(observation, dtype=np.float32)

    def reset(self):
        self.reward_shaping_total = None
        state = super().reset()
        self.actions = {}
        self.original_vec = self.__get_shooter_receiver_vec()
        self.holding_steps = 0
        self.stopped_steps = 0
        self.shooted = False
        return state

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    def _get_commands(self, actions):
        commands = []
        actions[1] = actions[1] if abs(actions[1]) > 0.5 else 0
        self.actions = actions
        cmd = Robot(yellow=False, id=0, v_x=0,
                    v_y=0, v_theta=actions[0],
                    kick_v_x=actions[1],
                    dribbler=True if actions[2] > 0 else False)
        commands.append(cmd)

        return commands

    def _calculate_reward_and_done(self):

        w_angle = 0.2
        w_ball_grad = 0.8
        reward = 0
        done = False
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {'pass_score': 0, 'angle': 0,
                                         'ball_grad': 0}
        if self.frame.robots_blue[1].infrared:
            reward += 1
            self.reward_shaping_total['pass_score'] += 1
            done = True
        else:
            rw_angle = w_angle * self.__angle_reward() if not self.shooted else 0
            rw_ball_grad = w_ball_grad * self.__ball_grad_rw()
            reward = rw_angle + rw_ball_grad
            self.reward_shaping_total['angle'] += rw_angle
            self.reward_shaping_total['ball_grad'] += rw_ball_grad
        if self.__wrong_ball() or self.holding_steps > 15:
            reward = -1
            done = True

        return reward, done

    def __get_shooter_receiver_vec(self):
        shooter_id = 0 if self.receiver_id else 1
        receiver = np.array([self.frame.robots_blue[self.receiver_id].x,
                             self.frame.robots_blue[self.receiver_id].y])
        shooter = np.array([self.frame.robots_blue[shooter_id].x,
                            self.frame.robots_blue[shooter_id].y])
        shooter_receiver = receiver - shooter
        shooter_receiver = shooter_receiver
        return shooter_receiver

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        pos_frame: Frame = Frame()
        def x(): return random.uniform(-1.5, 1.5)
        def y(): return random.uniform(1.5, -1.5)

        pos_frame.ball = Ball(x=x(), y=y())
        factor = (pos_frame.ball.y/abs(pos_frame.ball.y))
        offset = 0.115*factor
        angle = 270 if factor > 0 else 90
        pos_frame.robots_blue[0] = Robot(
            x=pos_frame.ball.x, y=pos_frame.ball.y+offset, theta=angle
        )
        shooter = np.array([pos_frame.ball.x, pos_frame.ball.y-0.1])
        receiver = np.array([x(), -pos_frame.ball.y])
        vect = receiver - shooter
        recv_angle = np.rad2deg(np.arctan2(vect[1], vect[0]) + np.pi)

        pos_frame.robots_blue[1] = Robot(x=receiver[0],
                                         y=receiver[1],
                                         theta=recv_angle)

        return pos_frame

    def __wrong_ball(self):
        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        last_ball = np.array([self.last_frame.ball.x, self.last_frame.ball.y])

        recv = np.array([self.frame.robots_blue[1].x,
                         self.frame.robots_blue[1].y])
        shooter = np.array([self.frame.robots_blue[0].x,
                            self.frame.robots_blue[0].y])
        comp_ball = np.array(ball*100, dtype=int)
        comp_shoot = np.array(shooter*100, dtype=int)
        comp_recv = np.array(recv*100, dtype=int)
        inside_x = min(comp_recv[0], comp_shoot[0]) <= comp_ball[0] <= max(comp_recv[0], comp_shoot[0])
        inside_y = min(comp_recv[1], comp_shoot[1]) <= comp_ball[1] <= max(comp_recv[1], comp_shoot[1])
        not_inside = not(inside_x and inside_y) and self.shooted
        last_dist = np.linalg.norm(last_ball - recv)
        dist = np.linalg.norm(ball - recv)
        stopped = abs(last_dist - dist) < 0.01
        if stopped:
            self.stopped_steps += 1
        else:
            self.stopped_steps = 0
        return self.stopped_steps > 20 or not_inside

    def __ball_grad_rw(self):
        assert(self.last_frame is not None)

        # Goal pos
        goal = np.array([self.frame.robots_blue[1].x,
                         self.frame.robots_blue[1].y])

        # Calculate previous ball dist
        last_ball = self.last_frame.ball
        ball = self.frame.ball
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_ball_dist = np.linalg.norm(goal - last_ball_pos)

        # Calculate new ball dist
        ball_pos = np.array([ball.x, ball.y])
        ball_dist = np.linalg.norm(goal - ball_pos)
        vect = self.__get_shooter_receiver_vec()
        ori_dist = np.linalg.norm(vect)

        ball_dist_rw= 0
        if self.shooted:
            ball_dist_rw = (ori_dist-ball_dist)/ori_dist

        if ball_dist_rw > 1:
            print("ball_dist -> ", ball_dist_rw)
            print(self.frame.ball)
            print(self.frame.robots_blue)
            print(self.frame.robots_yellow)
            print("===============================")

        return np.clip(ball_dist_rw, -1, 1)

    def __angle_reward(self):
        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        shooter = np.array([self.last_frame.robots_blue[0].x,
                            self.last_frame.robots_blue[0].y])

        ball = np.array([self.last_frame.ball.x, self.last_frame.ball.y])
        shooter_ball = ball - shooter
        dist_ball = np.linalg.norm(shooter_ball)
        shooter_ball /= dist_ball
        ori_norm = self.original_vec/np.linalg.norm(self.original_vec)
        cos_shooter = np.dot(shooter_ball, ori_norm)
        shooter_angle_reward = 0
        if not self.shooted:
            self.shooted = self.last_frame.robots_blue[0].infrared and self.actions[1] > 0
        if self.shooted:
            if np.rad2deg(np.arccos(cos_shooter)) > 1:
                shooter_angle_reward = -1
            else:
                shooter_angle_reward = 1
        elif self.last_frame.robots_blue[0].infrared:
            if self.holding_steps < 20:
                shooter_angle_reward = cos_shooter
            else:
                shooter_angle_reward -= 0.5
        return shooter_angle_reward - 1
