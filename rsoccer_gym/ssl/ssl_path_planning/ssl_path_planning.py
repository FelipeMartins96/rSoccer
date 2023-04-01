import random
from rsoccer_gym.Render.Render import RCGymRender

from rsoccer_gym.ssl.ssl_path_planning.navigation import Point2D, GoToPointEntry, RobotMove, go_to_point, abs_smallest_angle_diff, dist_to, length

import gym
import numpy as np
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree

ANGLE_TOLERANCE: float = np.deg2rad(7.5) # 7.5 degrees
SPEED_TOLERANCE: float = 0.20 # m/s == 20 cm/s
DIST_TOLERANCE: float = 0.10 # m == 10 cm
ANGULAR_SPEED_TOLERANCE: float = 5 # 5 rad/s

class SSLPathPlanningEnv(SSLBaseEnv):
    """The SSL robot needs to reach the target point with a given angle"""

    def __init__(self, field_type=1, n_robots_yellow=0):
        super().__init__(field_type=field_type, n_robots_blue=1,
                         n_robots_yellow=n_robots_yellow, time_step=0.025)

        self.action_space = gym.spaces.Box(low=-1, high=1,  # hyp tg.
                                           shape=(5, ), dtype=np.float32)

        n_obs = 6 + 4 + 7*self.n_robots_blue + 2*self.n_robots_yellow
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ),
                                                dtype=np.float32)

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10

        self.target_point: Point2D = Point2D(0, 0)
        self.target_angle: float = 0.0
        self.target_velocity: Point2D = Point2D(0, 0)

        self.reward_info = {
            'cumulative_dist_reward': 0,
            'cumulative_angle_reward': 0,
            'cumulative_velocity_reward': 0,
            'total_reward': 0,

            'dist_error': 0,
            'angle_error': 0,
            'velocity_error': 0,

            'current_speed': 0,
            'current_velocity_x': 0,
            'current_velocity_y': 0,
        }

        print('Environment initialized')
    
    def reset(self):
        self.reward_info = {
            'cumulative_dist_reward': 0,
            'cumulative_angle_reward': 0,
            'cumulative_velocity_reward': 0,
            'total_reward': 0,

            'dist_error': 0,
            'angle_error': 0,
            'velocity_error': 0,
            
            'current_speed': 0,
            'current_velocity_x': 0,
            'current_velocity_y': 0,
        }
        return super().reset()
    
    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_info

    def _frame_to_observations(self):
        observation = list()

        observation.append(self.norm_pos(self.target_point.x))
        observation.append(self.norm_pos(self.target_point.y))
        observation.append(np.sin(self.target_angle))
        observation.append(np.cos(self.target_angle))
        observation.append(self.norm_v(self.target_velocity.x))
        observation.append(self.norm_v(self.target_velocity.y))

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))

        return np.array(observation, dtype=np.float32)

    def convert_actions(self, move: RobotMove, angle):
        """Convert to local"""
        v_x = move.velocity.x
        v_y = move.velocity.y
        v_theta = move.angular_velocity

        # Convert to local
        v_x, v_y = v_x*np.cos(angle) + v_y*np.sin(angle),\
            -v_x*np.sin(angle) + v_y*np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x,v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c
        
        return v_x, v_y, v_theta

    def _get_commands(self, action):
        target_v_x = action[0] * self.max_v
        target_v_y = action[1] * self.max_v
        target_k_p = max(0, action[4] * 15)
        target_angle = target_k_p * np.arctan2(action[2], action[3])

        move = RobotMove(
            velocity=Point2D(target_v_x, target_v_y),
            angular_velocity=target_angle
        )

        v_x, v_y, v_theta = self.convert_actions(move, target_angle)

        return [
            Robot(
                yellow=False,
                id=0,
                v_x=v_x,
                v_y=v_y,
                v_theta=v_theta
            )
        ]

    def is_v_in_range(self, current, target) -> bool:
        return abs(current - target) <= SPEED_TOLERANCE

    def reward_function(self, robot_pos: Point2D, last_robot_pos: Point2D, robot_vel: Point2D, last_robot_vel: Point2D, robot_angle: float, target_pos: Point2D, target_angle: float, target_vel: Point2D):
        max_dist = np.sqrt(self.field.length ** 2 + self.field.width ** 2)

        last_dist_robot_to_target = dist_to(target_pos, last_robot_pos)
        dist_robot_to_target = dist_to(target_pos, robot_pos)

        last_robot_angle = np.deg2rad(self.last_frame.robots_blue[0].theta)
        last_angle_error = abs_smallest_angle_diff(last_robot_angle, target_angle)
        angle_error = abs_smallest_angle_diff(robot_angle, target_angle)

        last_robot_velocity_to_target = dist_to(target_vel, last_robot_vel)
        robot_velocity_to_target = dist_to(target_vel, robot_vel)

        angle_reward = 0.125 * (last_angle_error - angle_error) / np.pi
        dist_reward = 0.75 * (last_dist_robot_to_target - dist_robot_to_target) / max_dist
        velocity_reward = 0.125 * (last_robot_velocity_to_target - robot_velocity_to_target) / self.max_v
        
        angular_velocity = self.frame.robots_blue[0].v_theta
        angular_velocity_to_target = abs(angular_velocity)

        self.reward_info['dist_error'] = dist_robot_to_target
        self.reward_info['angle_error'] = angle_error
        self.reward_info['velocity_error'] = robot_velocity_to_target

        self.reward_info['current_speed'] = length(robot_vel)
        self.reward_info['current_velocity_x'] = robot_vel.x
        self.reward_info['current_velocity_y'] = robot_vel.y

        self.reward_info['total_reward'] += dist_reward
        self.reward_info['cumulative_dist_reward'] += dist_reward

        self.reward_info['total_reward'] += angle_reward
        self.reward_info['cumulative_angle_reward'] += angle_reward

        if dist_robot_to_target <= DIST_TOLERANCE:
            self.reward_info['total_reward'] += velocity_reward
            self.reward_info['cumulative_velocity_reward'] += velocity_reward

            if robot_velocity_to_target <= SPEED_TOLERANCE and angular_velocity_to_target <= ANGULAR_SPEED_TOLERANCE:
                return angle_reward, angle_error <= ANGLE_TOLERANCE

            return angle_reward + velocity_reward, False

        return dist_reward + angle_reward, False

    def _calculate_reward_and_done(self):
        robot = self.frame.robots_blue[0]
        last_robot = self.last_frame.robots_blue[0]

        robot_pos = Point2D(x=robot.x, y=robot.y)
        last_robot_pos = Point2D(x=last_robot.x, y=last_robot.y)
        robot_angle = np.deg2rad(robot.theta)
        target_pos = self.target_point
        target_angle = self.target_angle
        target_vel = self.target_velocity

        robot_vel = Point2D(x=robot.v_x, y=robot.v_y)
        last_robot_vel = Point2D(x=last_robot.v_x, y=last_robot.v_y)

        reward, done = self.reward_function(robot_pos=robot_pos,
                                            last_robot_pos=last_robot_pos,
                                            robot_vel=robot_vel,
                                            last_robot_vel=last_robot_vel,
                                            robot_angle=robot_angle,
                                            target_pos=target_pos,
                                            target_angle=target_angle,
                                            target_vel=target_vel)
        return reward, done

    def _get_initial_positions_frame(self):
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def get_random_x():
            return random.uniform(-field_half_length + 0.1,
                                  field_half_length - 0.1)

        def get_random_y():
            return random.uniform(-field_half_width + 0.1,
                                  field_half_width - 0.1)

        def get_random_theta():
            return random.uniform(0, 360)
        
        def get_random_speed():
            return random.uniform(0, 2.2)
        
        def get_random_w():
            return random.uniform(0, 9)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=get_random_x(), y=get_random_y())

        self.target_point = Point2D(x=get_random_x(), y=get_random_y())
        self.target_angle = np.deg2rad(get_random_theta())

        self.target_velocity = Point2D(0, 0) # Point2D(x=get_random_speed(), y=get_random_speed())

        #  TODO: Move RCGymRender to another place
        self.view = RCGymRender(self.n_robots_blue,
                                self.n_robots_yellow,
                                self.field,
                                simulator='ssl',
                                angle_tolerance=ANGLE_TOLERANCE)

        self.view.set_target(self.target_point.x, self.target_point.y)
        self.view.set_target_angle(np.rad2deg(self.target_angle))

        min_gen_dist = 0.2

        places = KDTree()
        places.insert((self.target_point.x, self.target_point.y))
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots_blue):
            pos = (get_random_x(), get_random_y())

            while places.get_nearest(pos)[1] < min_gen_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)

            speed = get_random_speed()
            vel_angle = get_random_theta()
            vel = (speed * np.cos(vel_angle), speed * np.sin(vel_angle))

            pos_frame.robots_blue[i] = Robot(id=i, yellow=False,
                                             x=pos[0], y=pos[1], v_x=vel[0], v_y=vel[1], v_theta=get_random_w(), theta=get_random_theta())

        for i in range(self.n_robots_yellow):
            pos = (get_random_x(), get_random_y())
            while places.get_nearest(pos)[1] < min_gen_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(id=i, yellow=True,
                                               x=pos[0], y=pos[1], theta=get_random_theta())

        return pos_frame
