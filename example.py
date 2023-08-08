import gym
import rsoccer_gym
import torch
import torch.nn as nn

class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
    
    def get_action(self, x):
        return self.net(x).detach().cpu().numpy()


# Using VSS Single Agent env
env = gym.make('VSSTNMT-v0')

cpt = torch.load('sa.pth')
pi = DDPGActor(cpt['N_OBS'], cpt['N_ACTS'])
pi.load_state_dict(state_dict=cpt['pi_state_dict'])

obs = env.reset()
# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    while not done:
        # Step using random actions
        action = env.action_space.sample() * 0
        action[5] = pi(torch.tensor(obs[5], dtype=torch.float32)).detach().numpy()

        obs, reward, done, _ = env.step(action)
        env.render()
    print(reward)