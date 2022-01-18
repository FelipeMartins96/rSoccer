import math
import random
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv
from rsoccer_gym.Utils import KDTree


class VSSGoToHRLEnv(VSSBaseEnv):
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

    def __init__(
        self,
        target_margin=0.0215,
        n_targets=1,
        n_blue_robots=1,
        n_yellow_robots=0,
        wor_w_energy=1e-3,
        wor_w_dist=1,
        man_w_goal = 10,
        man_w_ball_grad = 0.8,
        man_w_move = 0.2,
        man_w_energy = 2e-4,
        man_w_collision = 0,
    ):
        super().__init__(
            field_type=0,
            n_robots_blue=n_blue_robots,
            n_robots_yellow=n_yellow_robots,
            time_step=0.025,
            use_fira=False,
        )

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2, n_blue_robots*2), dtype=np.float32
        )
        n_obs = 4 + 7 * n_blue_robots + 5 * n_yellow_robots
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

        self.reward_shaping_total = None
        self.n_targets = n_targets * n_blue_robots
        # Initialize Class Atributes Manager
        self.man_w_goal = man_w_goal
        self.man_w_ball_grad = man_w_ball_grad
        self.man_w_move = man_w_move
        self.man_w_energy = man_w_energy
        self.man_w_collision = man_w_collision

        # Initialize Class Atributes Worker
        self.v_wheel_deadzone = 0.05
        self.target_margin = target_margin
        self.w_energy = wor_w_energy
        self.w_dist = wor_w_dist

        print("Environment initialized")

    def reset(self):
        self.reward_shaping_total = None
        self.targets = [(0., 0.)]
        self.previous_ball_potential = None
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
                self.n_robots_blue,
                self.n_robots_yellow,
                self.field,
                self.n_targets,
                simulator="vss",
            )

        return self.view.render_frame(
            self.frame, self.targets, return_rgb_array=mode == "rgb_array"
        )

    def _frame_to_observations(self):

        observation = []

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(np.sin(np.deg2rad(self.frame.robots_blue[i].theta)))
            observation.append(np.cos(np.deg2rad(self.frame.robots_blue[i].theta)))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
            observation.append(self.norm_w(self.frame.robots_yellow[i].v_theta))

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        # Manager Actions
        man_acts = actions[0]
        self.targets = [man_acts[i*2:i*2+2] * self.max_pos for i in range(self.n_robots_blue)]

        # Worker Actions
        commands = []
        for i in range(self.n_robots_blue):
            wor_acts = actions[1][i*2:i*2+2]
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(wor_acts)
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0, v_wheel1=v_wheel1))

        return commands

    def _calculate_reward_and_done(self):
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {
                "worker/dist": 0,
                "worker/weighted_dist": 0,
                "worker/energy": 0,
                "worker/weighted_energy": 0,
                "manager/move": 0,
                "manager/weighted_move": 0,
                "manager/ball_grad": 0,
                "manager/weighted_ball_grad": 0,
                "manager/energy": 0,
                "manager/weighted_energy": 0,
                "manager/collision": 0,
                "manager/weighted_collision": 0,
                "manager/goal_score": 0,
                "manager/goals_blue": 0,
                "manager/goals_yellow": 0,
            }

        reward = 0
        done = False

        # Worker Reward
        wor_reward = 0
        # Calculate distance as sum of distance of each sequential target
        dist = 0
        rbt = [self.frame.robots_blue[0].x, self.frame.robots_blue[0].y]
        p0 = rbt
        for t in self.targets:
            p1 = [t[0], t[1]]
            dist += np.linalg.norm(np.array(p1) - np.array(p0))
            p0 = p1

            # Distance reward
        dist_rw = -dist
        self.reward_shaping_total["worker/dist"] += dist_rw
        weighted_dist_rw = self.w_dist * dist_rw
        self.reward_shaping_total["worker/weighted_dist"] += weighted_dist_rw

        # Energy reward
        energy_pen = self.__energy_penalty()
        self.reward_shaping_total["worker/energy"] += energy_pen
        weighted_energy_pen = self.w_energy * energy_pen
        self.reward_shaping_total["worker/weighted_energy"] += weighted_energy_pen

        wor_reward += weighted_dist_rw + weighted_energy_pen

        # Manager Reward
        man_reward = 0
        goal = False

        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            self.reward_shaping_total["manager/goal_score"] += 1
            self.reward_shaping_total["manager/goals_blue"] += 1
            man_reward = 1 * self.man_w_goal
            goal = True
        elif self.frame.ball.x < -(self.field.length / 2):
            self.reward_shaping_total["manager/goal_score"] -= 1
            self.reward_shaping_total["manager/goals_yellow"] += 1
            man_reward = -1 * self.man_w_goal
            goal = True
        else:
            if self.last_frame is not None:
                # Calculate ball potential
                grad_ball_potential = self.__ball_grad()
                self.reward_shaping_total["manager/ball_grad"] += grad_ball_potential
                weighted_grad_ball_potential = (
                    self.man_w_ball_grad * grad_ball_potential
                )
                self.reward_shaping_total[
                    "manager/weighted_ball_grad"
                ] += weighted_grad_ball_potential

                # Calculate Move ball
                move_reward = self.__move_reward()
                self.reward_shaping_total["manager/move"] += move_reward
                weighted_move_reward = self.man_w_move * move_reward
                self.reward_shaping_total[
                    "manager/weighted_move"
                ] = weighted_move_reward

                # Calculate Energy penalty
                energy_penalty = self.__energy_penalty()
                self.reward_shaping_total["manager/energy"] += energy_penalty
                weighted_energy_penalty = self.man_w_energy * energy_penalty
                self.reward_shaping_total[
                    "manager/weighted_energy"
                ] = weighted_energy_penalty

                # Collision reward
                collision_pen = self.__collision_penalty()
                self.reward_shaping_total["manager/collision"] += collision_pen
                weighted_collision_pen = self.man_w_collision * collision_pen
                self.reward_shaping_total[
                    "manager/weighted_collision"
                ] = weighted_collision_pen

                man_reward = (
                    weighted_grad_ball_potential
                    + weighted_move_reward
                    + weighted_energy_pen
                    + weighted_collision_pen
                )

        done = goal

        return (man_reward, wor_reward), done

    def _get_initial_positions_frame(self):
        """Returns the position of each robot and ball for the initial frame"""
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x():
            return random.uniform(-field_half_length + 0.1, field_half_length - 0.1)

        def y():
            return random.uniform(-field_half_width + 0.1, field_half_width - 0.1)

        def theta():
            return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=x(), y=y())

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

        return left_wheel_speed, right_wheel_speed

    def __energy_penalty(self):
        """Calculates the energy penalty"""

        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = -(en_penalty_1 + en_penalty_2)
        return energy_penalty

    def __collision_penalty(self):

        collision_penalty = 0
        for yellow_rbt in self.frame.robots_yellow.values():
            rbt_speed = np.linalg.norm(np.array([yellow_rbt.v_x, yellow_rbt.v_y]))
            collision_penalty += -rbt_speed

        return collision_penalty

    def __ball_grad(self):
        '''Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        '''
        # Calculate ball potential
        length_cm = self.field.length * 100
        half_lenght = (self.field.length / 2.0)\
            + self.field.goal_depth

        # distance to defence
        dx_d = (half_lenght + self.frame.ball.x) * 100
        # distance to attack
        dx_a = (half_lenght - self.frame.ball.x) * 100
        dy = (self.frame.ball.y) * 100

        dist_1 = -math.sqrt(dx_a ** 2 + 2 * dy ** 2)
        dist_2 = math.sqrt(dx_d ** 2 + 2 * dy ** 2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        # Calculate ball potential gradient
        # = actual_potential - previous_potential
        if self.previous_ball_potential is not None:
            diff = ball_potential - self.previous_ball_potential
            grad_ball_potential = np.clip(diff * 3 / self.time_step,
                                          -5.0, 5.0)

        self.previous_ball_potential = ball_potential

        return grad_ball_potential

    def __move_reward(self):
        '''Calculate Move to ball reward

        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        '''

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        robot = np.array([self.frame.robots_blue[0].x,
                          self.frame.robots_blue[0].y])
        robot_vel = np.array([self.frame.robots_blue[0].v_x,
                              self.frame.robots_blue[0].v_y])
        robot_ball = ball - robot
        robot_ball = robot_ball/np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)

        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward
