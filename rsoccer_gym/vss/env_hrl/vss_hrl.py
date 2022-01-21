import gym
from rsoccer_gym.Simulators.rsim import RSimVSS


class VSSGoToHRLEnv(gym.Env):
    """
    VSS Environment with hierarchical structure
    """
    
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    NORM_BOUNDS = 1.2

    def __init__(
        self,
        field_type,
        n_robots_blue
        n_robots_yellow,
        time_step,
    ):
        self.sim = RSimVSS(
            field_type=field_type,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            time_step_ms=int(time_step * 1000),
        )
    
    def get_m_spaces(self):
        """
        Returns the observation space of the manager
        """
        pass
    
    def get_w_spaces(self):
        """
        Returns the observation space of the workers
        """
        pass
