# Unsupervised Skill Discovery using DIAYN + Soft Actor-Critic

An implementation of [**DIAYN** (Diversity Is All You Need)](https://arxiv.org/abs/1802.06070) built on top of [**Soft Actor-Critic (SAC)**](https://arxiv.org/abs/1801.01290) for unsupervised skill discovery in continuous control environments.

The agent learns a diverse set of distinguishable behaviours **without any extrinsic reward signal**, driven purely by an information-theoretic objective that maximises the mutual information between a latent skill variable and the states visited by the policy.

---

## Key Features

- **DIAYN pseudo-reward**: `r(s') = log q(z|s') - log p(z)` — rewards the agent for visiting states where the discriminator can identify the active skill
- **Soft Actor-Critic** with clipped double-Q, separate value network, and entropy-regularised policy
- **Vectorized training** via Gymnasium `SyncVectorEnv` (4 parallel envs) for ~2.5× wall-clock speedup
- **TensorBoard logging** with reward curves, discriminator accuracy, and per-skill histograms
- **Checkpoint save/resume** — interrupt and continue training seamlessly
- **Video export** of learned skills for qualitative evaluation

---

## Project Structure

```
robotics_project/
├── Brain/
│   ├── model.py            # PolicyNetwork, QvalueNetwork, ValueNetwork, Discriminator
│   ├── agent.py            # SACAgent — training loop, DIAYN reward, gradient updates
│   └── replay_memory.py    # Experience replay buffer (O(1) deque eviction, capacity 1M)
├── Common/
│   ├── config.py           # CLI argument parsing + default hyperparameters
│   ├── logger.py           # TensorBoard logging, checkpoint save/load, RAM monitoring
│   └── play.py             # Post-training skill evaluation with video recording
├── main.py                 # Entry point — vectorized training & evaluation
├── watch_live.py           # Live checkpoint visualization & MP4 export
├── high.py                 # Quick skill ranking script
├── requirements.txt        # Python dependencies
└── report.tex              # LaTeX project report
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/tennissta99660/robotics_project.git
cd robotics_project

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch ≥ 2.1
- Gymnasium[mujoco] ≥ 0.29.1
- NumPy, TensorBoard, OpenCV, psutil, tqdm

---

## Usage

### Training

```bash
# Train from scratch on Hopper-v4 with 50 skills
python main.py --env_name Hopper-v4 --n_skills 50 --do_train --train_from_scratch

# Resume training from last checkpoint
python main.py --env_name Hopper-v4 --n_skills 50 --do_train

# Train on a different environment
python main.py --env_name MountainCarContinuous-v0 --n_skills 50 --do_train --train_from_scratch
```

### Evaluation

```bash
# Evaluate all learned skills (saves videos to Vid/)
python main.py --env_name Hopper-v4 --n_skills 50
```

### Watch Live / Record Videos

```bash
# Record a random skill from the latest checkpoint
python watch_live.py --env_name Hopper-v4 --n_skills 50

# Record a specific skill
python watch_live.py --env_name Hopper-v4 --n_skills 50 --skill 3

# Record all skills
python watch_live.py --env_name Hopper-v4 --n_skills 50 --all_skills

# Live MuJoCo window
python watch_live.py --env_name Hopper-v4 --n_skills 50 --live
```

### Rank Skills by Reward

```bash
python high.py
```

### Monitor Training

```bash
tensorboard --logdir Logs/
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 3 × 10⁻⁴ (Adam) |
| Batch size | 256 |
| Discount factor (γ) | 0.99 |
| Entropy temperature (α) | 0.1 |
| Soft target update (τ) | 0.005 |
| Replay buffer size | 1,000,000 |
| Hidden layers | 2 × 512 (ReLU) |
| Number of skills | 50 |
| Max episodes | 10,000 |
| Parallel environments | 4 |

---

## Environments Tested

| Environment | State Dim | Action Dim |
|---|---|---|
| Hopper-v4 | 11 | 3 |
| BipedalWalker-v3 | 24 | 4 |
| MountainCarContinuous-v0 | 2 | 1 |
| MountainCar-v0 | 2 | 1 |

---

## References

1. B. Eysenbach, A. Gupta, J. Ibarz, S. Levine. *"Diversity is All You Need: Learning Skills without a Reward Function"*, ICLR 2019. [[arXiv]](https://arxiv.org/abs/1802.06070)
2. T. Haarnoja, A. Zhou, P. Abbeel, S. Levine. *"Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"*, ICML 2018. [[arXiv]](https://arxiv.org/abs/1801.01290)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
