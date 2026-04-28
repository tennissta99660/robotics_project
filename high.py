import gymnasium as gym
import numpy as np
import torch, glob
from Brain import SACAgent

env_name='Humanoid-v5'; n_skills=40
env = gym.make(env_name)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
action_bounds = [env.action_space.low[0], env.action_space.high[0]]
p_z = np.full(n_skills, 1/n_skills)
agent = SACAgent(p_z=p_z, n_states=n_states, n_actions=n_actions, action_bounds=action_bounds, n_skills=n_skills, lr=3e-4, batch_size=256, max_n_episodes=1, max_episode_len=1000, gamma=0.99, alpha=0.1, tau=0.005, n_hiddens=512, mem_size=1000, reward_scale=1, seed=123, env_name=env_name, interval=10, do_train=False, train_from_scratch=False, num_envs=1)
agent.set_policy_net_to_cpu_mode()
agent.set_policy_net_to_eval_mode()
dirs = sorted(glob.glob('Checkpoints/Humanoid/*/'))
c = torch.load(dirs[-1]+'params.pth', weights_only=False, map_location='cpu')
agent.policy_network.load_state_dict(c['policy_network_state_dict'])
ep = c.get('episode', '?')
print(f'Checkpoint episode: {ep}')
results = []
for z in range(n_skills):
    s, _ = env.reset(seed=42)
    oh = np.zeros(n_skills); oh[z]=1; s = np.concatenate([s, oh])
    r_total = 0
    for _ in range(1000):
        a = agent.choose_action(s)
        s_, r, term, trunc, _ = env.step(a)
        oh = np.zeros(n_skills); oh[z]=1; s_ = np.concatenate([s_, oh])
        r_total += r
        if term or trunc: break
        s = s_
    results.append((z, r_total))
    print(f'Skill {z:2d}: {r_total:.1f}')
results.sort(key=lambda x: -x[1])
print('\n=== TOP 10 SKILLS ===')
for rank, (z, r) in enumerate(results[:10], 1):
    print(f'{rank}. Skill {z:2d}  reward: {r:.1f}')
env.close()