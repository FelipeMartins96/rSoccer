from gym.envs.registration import register

register(
        id='VSSHRL-v0',
        entry_point='rsoccer_gym.vss.env_hrl:VSSHRLEnv',
        kwargs={'field_type': 0, 'n_robots_blue': 1, 'n_robots_yellow': 0, 'time_step': 0.025},
        max_episode_steps=600,
)
register(
    id='VSSHRL-v1',
    entry_point='rsoccer_gym.vss.env_hrl:VSSHRLEnv',
    kwargs={
        'field_type': 0,
        'n_robots_blue': 1,
        'n_robots_yellow': 0,
        'time_step': 0.025,
        'm_w_goal': 1,
        'm_w_ball_grad': 1/0.75,
        'm_w_move': 0,
        'm_w_energy': 0,
        'm_w_collision': 0,
        'w_w_dist': 1/2.15,
        'w_w_energy': 0,
        'v_wheel_deadzone': 0.05,
    },
    max_episode_steps=600,
)
register(
    id='VSSHRL-v2',
    entry_point='rsoccer_gym.vss.env_hrl:VSSHRLEnv',
    kwargs={
        'field_type': 0,
        'n_robots_blue': 1,
        'n_robots_yellow': 0,
        'time_step': 0.025,
        'm_w_goal': 1,
        'm_w_ball_grad': 1/0.75,
        'm_w_move': 1/120,
        'm_w_energy': 1/40000,
        'm_w_collision': 0,
        'w_w_dist': 1/2.15,
        'w_w_energy': 0,
        'v_wheel_deadzone': 0.05,
    },
    max_episode_steps=600,
)
register(
    id='VSSHRL-v3',
    entry_point='rsoccer_gym.vss.env_hrl:VSSHRLEnv',
    kwargs={
        'field_type': 0,
        'n_robots_blue': 1,
        'n_robots_yellow': 0,
        'time_step': 0.025,
        'm_w_goal': 1,
        'm_w_ball_grad': 1/0.75,
        'm_w_move': 1,
        'm_w_energy': 1/40000,
        'm_w_collision': 0,
        'w_w_dist': 1/2.15,
        'w_w_energy': 0,
        'v_wheel_deadzone': 0.05,
    },
    max_episode_steps=600,
)
register(
    id='VSSHRL-v4',
    entry_point='rsoccer_gym.vss.env_hrl:VSSHRLEnv',
    kwargs={
        'field_type': 0,
        'n_robots_blue': 1,
        'n_robots_yellow': 0,
        'time_step': 0.025,
        'm_w_goal': 1,
        'm_w_ball_grad': 1/0.75,
        'm_w_move': 2,
        'm_w_energy': 1/40000,
        'm_w_collision': 0,
        'w_w_dist': 1/2.15,
        'w_w_energy': 0,
        'v_wheel_deadzone': 0.05,
    },
    max_episode_steps=600,
)
register(
    id='VSSHRL-v5',
    entry_point='rsoccer_gym.vss.env_hrl:VSSHRLEnv',
    kwargs={
        'field_type': 0,
        'n_robots_blue': 1,
        'n_robots_yellow': 0,
        'time_step': 0.025,
        'm_w_goal': 1,
        'm_w_ball_grad': 1/0.75,
        'm_w_move': 1/120,
        'm_w_energy': 1/40000,
        'm_w_collision': 0,
        'w_w_dist': 1/2.15,
        'w_w_energy': 0,
        'v_wheel_deadzone': 0.05,
    },
    max_episode_steps=600,
)

register(id='VSSGoTo-v0',
         entry_point='rsoccer_gym.vss.env_goto:VSSGoToEnv',
        kwargs={'n_targets': 1, 'w_energy': 0},
         max_episode_steps=600
         )
register(id='VSSGoTo-v1',
         entry_point='rsoccer_gym.vss.env_goto:VSSGoToEnv',
        kwargs={'n_targets': 3, 'w_energy': 0},
         max_episode_steps=600
         )
register(id='VSSGoTo-v2',
         entry_point='rsoccer_gym.vss.env_goto:VSSGoToEnv',
        kwargs={'n_targets': 1, 'n_blue_robots': 1, 'n_yellow_robots': 5, 'w_energy': 0},
         max_episode_steps=600
         )
register(id='VSSGoTo-v3',
         entry_point='rsoccer_gym.vss.env_goto:VSSGoToEnv',
        kwargs={'n_targets': 3, 'n_blue_robots': 1, 'n_yellow_robots': 5, 'w_energy': 0},
         max_episode_steps=600
         )

register(id='VSSGoToHRL-v0',
         entry_point='rsoccer_gym.vss.env_goto:VSSGoToHRLEnv',
        kwargs={'n_targets': 1, 'n_blue_robots': 1, 'n_yellow_robots': 0, 'wor_w_energy': 0},
         max_episode_steps=1200
         )

register(id='VSSGoToHRL-v1',
         entry_point='rsoccer_gym.vss.env_goto:VSSGoToHRLEnv',
        kwargs={'n_targets': 1, 'n_blue_robots': 2, 'n_yellow_robots': 0, 'wor_w_energy': 0},
         max_episode_steps=1200
         )


register(id='VSS-v0',
         entry_point='rsoccer_gym.vss.env_vss:VSSEnv',
         max_episode_steps=1200
         )

register(id='VSSMA-v0',
         entry_point='rsoccer_gym.vss.env_ma:VSSMAEnv',
         max_episode_steps=1200
         )

register(id='VSSMAOpp-v0',
         entry_point='rsoccer_gym.vss.env_ma:VSSMAOpp',
         max_episode_steps=1200
         )

register(id='VSSGk-v0',
         entry_point='rsoccer_gym.vss.env_gk:rSimVSSGK',
         max_episode_steps=1200
         )

register(id='VSSFIRA-v0',
         entry_point='rsoccer_gym.vss.env_vss:VSSEnv',
         kwargs={'use_fira': True},
         max_episode_steps=1200
         )

register(id='SSLGoToBall-v0',
         entry_point='rsoccer_gym.ssl.ssl_go_to_ball:SSLGoToBallEnv',
         kwargs={'field_type': 2, 'n_robots_yellow': 6},
         max_episode_steps=1200
         )

register(id='SSLGoToBallIR-v0',
         entry_point='rsoccer_gym.ssl.ssl_go_to_ball:SSLGoToBallIREnv',
         kwargs={'field_type': 2, 'n_robots_yellow': 6},
         max_episode_steps=1200
         )

register(id='SSLGoToBallShoot-v0',
         entry_point='rsoccer_gym.ssl.ssl_go_to_ball_shoot:SSLGoToBallShootEnv',
         kwargs={'field_type': 2, 'random_init': True,
                 'enter_goal_area': False},
         max_episode_steps=2400
         )

register(id='SSLStaticDefenders-v0',
         entry_point='rsoccer_gym.ssl.ssl_hw_challenge.static_defenders:SSLHWStaticDefendersEnv',
         kwargs={'field_type': 2},
         max_episode_steps=1000
         )

register(id='SSLDribbling-v0',
         entry_point='rsoccer_gym.ssl.ssl_hw_challenge.dribbling:SSLHWDribblingEnv',
         max_episode_steps=4800
         )

register(id='SSLContestedPossession-v0',
         entry_point='rsoccer_gym.ssl.ssl_hw_challenge.contested_possession:SSLContestedPossessionEnv',
         max_episode_steps=1200
         )

register(id='SSLPassEndurance-v0',
         entry_point='rsoccer_gym.ssl.ssl_hw_challenge:SSLPassEnduranceEnv',
         max_episode_steps=120
         )

register(id='SSLPassEnduranceMA-v0',
         entry_point='rsoccer_gym.ssl.ssl_hw_challenge:SSLPassEnduranceMAEnv',
         max_episode_steps=1200
         )
