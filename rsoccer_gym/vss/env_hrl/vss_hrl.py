import gym
import numpy as np
import jax
from rsoccer_gym.Simulators.rsim import RSimVSS
from rsoccer_gym.Entities import Frame, Ball, Robot
from rsoccer_gym.Utils import KDTree
from collections import namedtuple

Observations = namedtuple("Observations", ["ball", "blue", "yellow"])


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
    ):
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

    def step(self, w_actions):
        """
        Performs one step of the environment
        """
        assert self.is_set_action_m, 'Manager action not set'

        # Denormalize actions, transform to v_wheel speeds
        commands = self._get_commands(w_actions)
        self.sim.send_commands(commands)

        # Get Frame from Simulator
        frame = self.sim.get_frame()

        # Get observations
        obs = self._frame_to_observations(frame)

        # Check for terminal state
        done = is_done(frame)

        # Calculate rewards
        rewards = self._get_rewards(frame) if not done else 0

        # Reset manager action flag
        self.is_set_action_m = False

        return obs, rewards, done, {}

    def reset(self):
        """
        Resets the environment
        """
        self.is_set_action_m = False
        self.sim.reset(self._get_initial_frame())
        frame = self.sim.get_frame()
        observations = self._frame_to_observations(frame)
        m_obs = self.get_obs_m(observations)
        return m_obs

    def _frame_to_observations(self, frame):
        """
        Converts the frame to observations
        """
        # TODO: get norm arrays
        self.ball_norm = np.array([0.1, 0.1, 0.1, 0.1])
        self.blue_norm = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.yellow_norm = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        ball_np = np.array([frame.ball.x, frame.ball.y, frame.ball.v_x, frame.ball.v_y])
        blue_np = np.array(
            [
                [
                    frame.robots_blue[i].x,
                    frame.robots_blue[i].y,
                    np.sin(np.deg2rad(frame.robots_blue[i].theta)),
                    np.cos(np.deg2rad(frame.robots_blue[i].theta)),
                    frame.robots_blue[i].v_x,
                    frame.robots_blue[i].v_y,
                    frame.robots_blue[i].v_theta,
                ]
                for i in range(self.n_robots_blue)
            ]
        )
        yellow_np = np.array(
            [
                [
                    frame.robots_yellow[i].x,
                    frame.robots_yellow[i].y,
                    frame.robots_yellow[i].v_x,
                    frame.robots_yellow[i].v_y,
                    frame.robots_yellow[i].v_theta,
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
        h_width = self.field.width / 2 - min_dist
        h_length = self.field.length / 2 - min_dist

        def randomize_pos(key):
            new_k, x_k, y_k, theta_k = jax.random.split(key, 4)
            x = jax.random.uniform(x_k, minval=-h_width, maxval=h_width)
            y = jax.random.uniform(y_k, minval=-h_length, maxval=h_length)
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
