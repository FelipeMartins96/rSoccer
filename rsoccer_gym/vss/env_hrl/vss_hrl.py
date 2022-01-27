import gym
import numpy as np
import jax
from rsoccer_gym.Simulators.rsim import RSimVSS
from rsoccer_gym.Entities import Frame, Ball, Robot
from rsoccer_gym.Utils import KDTree
from collections import namedtuple
import math

Observations = namedtuple("Observations", ["ball", "blue", "yellow"])
Rewards = namedtuple("Rewards", ["manager", "workers"])
Obs = namedtuple("Obs", ['manager', 'workers'])


class VSSHRLEnv(gym.Env):
    """
    VSS Environment with hierarchical structure
    One manager sends the target pos for every worker
    Every worker uses the same network, and observes its position and the target pos
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    NORM_BOUNDS = 1.2

    def __init__(
        self,
        field_type,
        n_robots_blue,
        n_robots_yellow,
        time_step,
        m_w_goal=10,
        m_w_ball_grad=40,
        m_w_move=0,
        m_w_energy=0,
        m_w_collision=0,
        w_w_dist=1,
        w_w_energy=0,
        v_wheel_deadzone=0.05,
    ):
        # Environment recieves action for all robots
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=((n_robots_blue + n_robots_yellow) * 2,),
            dtype=np.float32,
        )
        self.sim = RSimVSS(
            field_type=field_type,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            time_step_ms=int(time_step * 1000),
        )

        # Initialize Attributes
        self.field = self.sim.get_field_params()
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.n_controlled_robots = n_robots_blue
        self.v_wheel_deadzone = v_wheel_deadzone
        self.view = None
        self.max_pos, self.max_v, self.max_w = self._get_sim_maximuns()
        self.ball_norm, self.blue_norm, self.yellow_norm = self._get_obs_norms()

        self.m_reward_weights = np.array(
            [m_w_goal, m_w_ball_grad, m_w_move, m_w_collision, m_w_energy]
        )
        self.w_reward_weights = np.array([w_w_dist, w_w_energy])

    def reset(self):
        """
        Resets the environment
        """
        self.is_set_action_m = False
        self.targets = None
        self.info = None
        self.sim.reset(self._get_initial_frame())
        self.frame = self.sim.get_frame()
        self.observations = self._frame_to_observations()
        return self._get_obs_m()

    def step(self, w_actions):
        """
        Performs one step of the environment
        """
        assert self.is_set_action_m, 'Manager action not set'

        # Denormalize actions, transform to v_wheel speeds
        commands = self._get_commands(w_actions)
        self.sent_commands = commands
        self.sim.send_commands(commands)

        # Get Frame from Simulator
        self.last_frame = self.frame
        self.frame = self.sim.get_frame()

        # Get observations
        self.observations = self._frame_to_observations()

        # Calculate rewards
        m_reward, w_rewards, done = self._get_rewards_and_done()

        # Reset manager action flag
        self.is_set_action_m = False

        return (
            Obs(self._get_obs_m(), self._get_obs_w()),
            Rewards(m_reward, w_rewards),
            done,
            self.info,
        )

    def render(self, mode='human'):
        if self.view is None:
            from rsoccer_gym.Render import RCGymRender

            self.view = RCGymRender(
                self.n_robots_blue, self.n_robots_yellow, self.field, simulator='vss'
            )

        return self.view.render_frame(
            self.sim.get_frame(),
            None if self.targets is None else self.targets * self.max_pos,
            return_rgb_array=mode == 'rgb_array',
        )

    def get_spaces_m(self):
        """
        Returns the observation space of the manager
        """
        # Manager action are (X, Y) target for each controlled robot
        # The order is blue robots in crescent id order followed by yellow robots
        m_action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=((self.n_controlled_robots * 2,)),
            dtype=np.float32,
        )

        # Manager observation is the global position of the ball and every robot
        # the ball has 4 obs, the blue robots each have 7 obs, and yellow robots have 5
        m_observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(4 + (self.n_robots_blue * 7) + (self.n_robots_yellow * 5),),
            dtype=np.float32,
        )

        return m_observation_space, m_action_space

    def get_spaces_w(self):
        """
        Returns the observation space of the workers
        """
        # The environment will require actions even for non controlled robots

        # Worker action are left and right wheel speeds
        w_action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32,
        )

        # Worker observation is its own global position and the target position
        w_observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(7 + 2,),
            dtype=np.float32,
        )

        return w_observation_space, w_action_space

    def set_key(self, key):
        self.key = key

    def set_action_m(self, action):
        """
        Sets the action of the manager
        """
        assert not self.is_set_action_m, 'Manager action already set'

        self.targets = action.reshape((-1, 2))
        self.is_set_action_m = True

        return self._get_obs_w()

    def _get_rewards_and_done(self):
        # Manager Rewards
        m_rewards = np.array(
            [
                self._rw_goal(),
                self._rw_ball_grad(),
                self._rw_move(),
                self._rw_collision(),
                sum([self._rw_energy(i) for i in range(self.n_controlled_robots)]),
            ]
        )

        # Workers Rewards
        w_rewards = np.array(
            [
                [self._rw_dist(i), self._rw_energy(i)]
                for i in range(self.n_controlled_robots)
            ]
        )

        m_weighted_rewards = m_rewards * self.m_reward_weights
        w_weighted_rewards = w_rewards * self.w_reward_weights

        if self.info is None:
            self.info = {
                'manager_rw': m_rewards,
                'workers_rw': w_rewards,
                'manager_weighted_rw': m_weighted_rewards,
                'workers_weighted_rw': w_weighted_rewards,
            }
        else:
            self.info['manager_rw'] += m_rewards
            self.info['workers_rw'] += w_rewards
            self.info['manager_weighted_rw'] += m_weighted_rewards
            self.info['workers_weighted_rw'] += w_weighted_rewards

        done = True if m_rewards[0] != 0 else False

        return m_weighted_rewards.sum(), w_weighted_rewards.sum(axis=1), done

    def _get_sim_maximuns(self):
        max_pos = max(
            self.field.width / 2, (self.field.length / 2) + self.field.penalty_length
        )
        max_wheel_rad_s = (self.field.rbt_motor_max_rpm / 60) * 2 * np.pi
        max_v = max_wheel_rad_s * self.field.rbt_wheel_radius
        # 0.04 = robot radius (0.0375) + wheel thicknees (0.0025)
        max_w = np.rad2deg(max_v / 0.04)

        return max_pos, max_v, max_w

    def _get_commands(self, acts):
        # Denormalize
        dn_acts = np.clip(acts, -1, 1) * self.max_v
        # Process deadzone
        pc_acts = dn_acts * (
            (dn_acts < -self.v_wheel_deadzone) | (dn_acts > self.v_wheel_deadzone)
        )
        actions = pc_acts.reshape((-1, 2)) / self.field.rbt_wheel_radius

        commands = []
        for i in range(self.n_robots_blue):
            v_wheel0, v_wheel1 = actions[i]
            commands.append(
                Robot(yellow=False, id=i, v_wheel0=v_wheel0, v_wheel1=v_wheel1)
            )

        for i in range(self.n_robots_yellow):
            j = i + self.n_robots_blue
            v_wheel0, v_wheel1 = actions[j]
            commands.append(
                Robot(yellow=True, id=i, v_wheel0=v_wheel0, v_wheel1=v_wheel1)
            )

        return commands

    def _get_obs_m(self):
        """
        Returns the observations of the manager
        """
        return np.concatenate(
            [
                self.observations.ball,
                self.observations.blue.flatten(),
                self.observations.yellow.flatten(),
            ]
        )

    def _get_obs_w(self):
        return np.stack(
            [
                np.concatenate([self.observations.blue[i], self.targets[i]])
                for i in range(self.n_controlled_robots)
            ]
        )

    def _frame_to_observations(self):
        """
        Converts the frame to observations
        """

        ball_np = np.array(
            [
                self.frame.ball.x,
                self.frame.ball.y,
                self.frame.ball.v_x,
                self.frame.ball.v_y,
            ]
        )
        blue_np = np.array(
            [
                [
                    self.frame.robots_blue[i].x,
                    self.frame.robots_blue[i].y,
                    np.sin(np.deg2rad(self.frame.robots_blue[i].theta)),
                    np.cos(np.deg2rad(self.frame.robots_blue[i].theta)),
                    self.frame.robots_blue[i].v_x,
                    self.frame.robots_blue[i].v_y,
                    self.frame.robots_blue[i].v_theta,
                ]
                for i in range(self.n_robots_blue)
            ]
        )
        yellow_np = np.array(
            [
                [
                    self.frame.robots_yellow[i].x,
                    self.frame.robots_yellow[i].y,
                    self.frame.robots_yellow[i].v_x,
                    self.frame.robots_yellow[i].v_y,
                    self.frame.robots_yellow[i].v_theta,
                ]
                for i in range(self.n_robots_yellow)
            ]
        )
        ball_norm = ball_np * self.ball_norm
        blue_norm = blue_np * self.blue_norm
        yellow_norm = (
            yellow_np * self.yellow_norm if self.n_robots_yellow else yellow_np
        )

        return Observations(ball=ball_norm, blue=blue_norm, yellow=yellow_norm)

    def _get_initial_frame(self):
        """
        Returns the initial frame
        """
        assert hasattr(self, 'key'), 'Jax key not set'

        min_dist = 0.1
        h_length = self.field.length / 2 - min_dist
        h_width = self.field.width / 2 - min_dist

        def randomize_pos(key):
            new_k, x_k, y_k, theta_k = jax.random.split(key, 4)
            x = jax.random.uniform(x_k, minval=-h_length, maxval=h_length)
            y = jax.random.uniform(y_k, minval=-h_width, maxval=h_width)
            theta = jax.random.uniform(theta_k) * 360

            return new_k, x, y, theta

        frame = Frame()

        # Randomize ball position
        self.key, x, y, theta = randomize_pos(self.key)
        frame.ball = Ball(x=x, y=y)

        tree = KDTree()
        pos = (x, y)
        tree.insert(pos)

        # Randomize blue robots
        for i in range(self.n_robots_blue):
            while tree.get_nearest(pos)[1] < min_dist:
                self.key, x, y, theta = randomize_pos(self.key)
                pos = (x, y)
            tree.insert(pos)
            frame.robots_blue[i] = Robot(x=x, y=pos[1], theta=theta)

        # Randomize yellow robots
        for i in range(self.n_robots_yellow):
            while tree.get_nearest(pos)[1] < min_dist:
                self.key, x, y, theta = randomize_pos(self.key)
                pos = (x, y)
            tree.insert(pos)
            frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta)

        return frame

    def _get_obs_norms(self):
        """
        Returns the normalization arrays of the observations
        """
        pos_factor = 1 / self.max_pos
        v_factor = 1 / self.max_v
        w_factor = 1 / self.max_w

        # 4 obs: X, Y, V_X, V_Y
        ball_norm = np.array([pos_factor, pos_factor, v_factor, v_factor])
        # 7 obs: X, Y, cos(THETA), sin(THETA), V_X, V_Y, V_THETA
        blue_norm = np.array(
            [pos_factor, pos_factor, 1, 1, v_factor, v_factor, w_factor]
        )
        # 5 obs: X, Y, V_X, V_Y, V_THETA
        yellow_norm = np.array([pos_factor, pos_factor, v_factor, v_factor, w_factor])

        return ball_norm, blue_norm, yellow_norm

    # -------------------Rewards-------------------

    def _rw_goal(self):
        """
        1 if goal blue, -1 if goal yellow, 0 if not goal
        """
        if self.frame.ball.x > self.field.length / 2:
            return 1
        elif self.frame.ball.x < -self.field.length / 2:
            return -1
        return 0

    def _rw_ball_grad(self):
        """
        ball grad is given by the distance delta from the ball to the yellow goal
        """
        last_ball = np.array([self.last_frame.ball.x, self.last_frame.ball.y])
        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        yellow_goal = np.array([self.field.length / 2, 0])

        last_dist = np.linalg.norm(yellow_goal - last_ball)
        dist = np.linalg.norm(yellow_goal - ball)

        return last_dist - dist

    def _rw_move(self):
        """
        move is given by min dist between blue robots and ball
        """
        last_ball = np.array([self.last_frame.ball.x, self.last_frame.ball.y])
        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        move_min = None
        for rbt_id in self.frame.robots_blue:
            rbt = np.array(
                [self.frame.robots_blue[rbt_id].x, self.frame.robots_blue[rbt_id].y]
            )
            last_rbt = np.array(
                [
                    self.last_frame.robots_blue[rbt_id].x,
                    self.last_frame.robots_blue[rbt_id].y,
                ]
            )
            dist = np.linalg.norm([rbt, ball])
            last_dist = np.linalg.norm([last_rbt, last_ball])
            move = last_dist - dist
            move_min = move if move_min is None or move < move_min else move_min

        return move_min

    def _rw_collision(self):
        """
        collision is given by the yellow robots velocity
        """
        collision_penalty = 0
        for yellow_rbt in self.frame.robots_yellow.values():
            rbt_speed = np.linalg.norm(np.array([yellow_rbt.v_x, yellow_rbt.v_y]))
            collision_penalty -= rbt_speed

        return collision_penalty

    def _rw_energy(self, id):
        """
        energy is given the sum of sent wheel speed
        """
        en_wheel0 = abs(self.sent_commands[id].v_wheel0)
        en_wheel1 = abs(self.sent_commands[id].v_wheel1)
        return -(en_wheel0 + en_wheel1)

    def _rw_dist(self, id):
        """
        dist is given by the distance between the robot and the target
        """
        rbt = np.array([self.frame.robots_blue[id].x, self.frame.robots_blue[id].y])
        target = self.targets[id]
        dist = np.linalg.norm(rbt - target)
        return -dist
