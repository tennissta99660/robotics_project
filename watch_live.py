"""
Watch the agent by loading the latest checkpoint and recording a short video.
Run this in a SEPARATE terminal while training continues.

Usage:
    python watch_live.py                              # record 1 random skill
    python watch_live.py --skill 3                    # record skill 3
    python watch_live.py --all_skills                 # record all skills
    python watch_live.py --live                       # live MuJoCo window (laggy)
"""

import gymnasium as gym
import numpy as np
import torch
import glob
import os
import argparse
import cv2

from Brain import SACAgent
from Common import get_params


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


def load_latest_checkpoint(env_name, agent):
    """Load the most recently saved checkpoint for the given env."""
    env_base = env_name.rsplit("-", 1)[0]
    model_dirs = glob.glob(f"Checkpoints/{env_base}/*/")
    model_dirs.sort()
    if not model_dirs:
        path = f"Checkpoints/{env_base}/params.pth"
        if os.path.exists(path):
            checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        else:
            raise FileNotFoundError(f"No checkpoint found for {env_name}")
    else:
        checkpoint = torch.load(model_dirs[-1] + "params.pth",
                                weights_only=False, map_location="cpu")

    agent.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
    agent.value_network.load_state_dict(checkpoint["value_network_state_dict"])
    agent.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    episode = checkpoint.get("episode", "?")
    print(f"Loaded checkpoint at episode {episode}")
    return episode


def run_skill(env, agent, skill, n_skills, max_steps=1000, record=False):
    """Run one episode with a specific skill. Returns (reward, frames)."""
    s, _ = env.reset()
    s = concat_state_latent(s, skill, n_skills)
    total_reward = 0
    frames = []

    for step in range(max_steps):
        action = agent.choose_action(s)
        s_, reward, terminated, truncated, _ = env.step(action)
        s_ = concat_state_latent(s_, skill, n_skills)
        total_reward += reward

        if record:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if terminated or truncated:
            break
        s = s_

    return total_reward, frames


def save_video(frames, filename, fps=30):
    """Save frames to an mp4 video file."""
    if not frames:
        print("No frames to save!")
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved: {filename}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch agent from checkpoint")
    parser.add_argument("--env_name", default="Humanoid-v5", type=str)
    parser.add_argument("--n_skills", default=40, type=int)
    parser.add_argument("--skill", default=None, type=int,
                        help="Specific skill to watch. Random if not set.")
    parser.add_argument("--all_skills", action="store_true",
                        help="Record all skills.")
    parser.add_argument("--live", action="store_true",
                        help="Use live MuJoCo window (laggy during training).")
    args = parser.parse_args()

    # Pick render mode
    render_mode = "human" if args.live else "rgb_array"
    env = gym.make(args.env_name, render_mode=render_mode)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]

    # Build agent
    p_z = np.full(args.n_skills, 1 / args.n_skills)
    agent = SACAgent(
        p_z=p_z,
        n_states=n_states,
        n_actions=n_actions,
        action_bounds=action_bounds,
        n_skills=args.n_skills,
        lr=3e-4, batch_size=256, max_n_episodes=1, max_episode_len=1000,
        gamma=0.99, alpha=0.1, tau=0.005, n_hiddens=512,
        mem_size=1000, reward_scale=1, seed=123,
        env_name=args.env_name, interval=10, do_train=False,
        train_from_scratch=False, num_envs=1,
    )
    agent.set_policy_net_to_cpu_mode()
    agent.set_policy_net_to_eval_mode()

    # Load checkpoint
    episode = load_latest_checkpoint(args.env_name, agent)

    # Decide which skills to run
    if args.all_skills:
        skills_to_run = list(range(args.n_skills))
    elif args.skill is not None:
        skills_to_run = [args.skill]
    else:
        skills_to_run = [np.random.choice(args.n_skills)]

    os.makedirs("live_vid", exist_ok=True)

    for z in skills_to_run:
        print(f"\n>>> Skill {z} | checkpoint ep {episode}")
        reward, frames = run_skill(env, agent, z, args.n_skills,
                                   record=(not args.live))

        if not args.live and frames:
            filename = f"live_vid/ep{episode}_skill{z}.mp4"
            save_video(frames, filename)

        print(f"    Reward: {reward:.1f}")

    env.close()
    if not args.live:
        print(f"\nDone! Videos saved in Vid/ — open them to watch.")
