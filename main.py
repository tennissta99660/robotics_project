import gymnasium as gym
from Brain import SACAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helper: concatenate state with one-hot skill vector
# Supports both single state (1D) and batch of states (2D).
# ---------------------------------------------------------------------------
def concat_state_latent(s, z_, n):
    """
    s   : np.array of shape (n_states,) or (num_envs, n_states)
    z_  : int  or  np.array of shape (num_envs,)
    n   : number of skills
    """
    if s.ndim == 1:
        # Single env — original behaviour
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])
    else:
        # Vectorized: s is (num_envs, n_states), z_ is (num_envs,)
        num_envs = s.shape[0]
        z_one_hot = np.zeros((num_envs, n))
        z_one_hot[np.arange(num_envs), z_] = 1   # fancy-index assignment, O(num_envs)
        return np.concatenate([s, z_one_hot], axis=1)


if __name__ == "__main__":
    params = get_params()
    num_envs = params["num_envs"]

    # ------------------------------------------------------------------
    # Probe a single env to get space dimensions (same as before)
    # ------------------------------------------------------------------
    test_env = gym.make(params["env_name"])
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
    test_env.close()

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)

    # ------------------------------------------------------------------
    # Build agent and logger (unchanged)
    # ------------------------------------------------------------------
    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)

    # ==================================================================
    # TRAINING
    # ==================================================================
    if params["do_train"]:

        # ---- Resume or scratch ----------------------------------------
        if not params["train_from_scratch"]:
            episode, last_logq_zs, np_rng_state, torch_rng_state, random_rng_state = logger.load_weights()
            agent.hard_update_target_network()
            min_episode = episode
            np.random.set_state(np_rng_state)
            agent.set_rng_states(torch_rng_state, random_rng_state)
            print("Keep training from previous run.")
        else:
            min_episode = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            print("Training from scratch.")

        # ---- Build vectorized environment -----------------------------
        # "sync" mode: envs step sequentially inside one process.
        # MuJoCo's C library releases the GIL so threads run in parallel;
        # sync is simpler and more stable on Kaggle than async.
        vec_env = gym.make_vec(
            params["env_name"],
            num_envs=num_envs,
            vectorization_mode="sync",
        )
        vec_env.reset(seed=params["seed"])

        # ---- Per-env tracking state -----------------------------------
        # Each parallel env independently tracks its current episode.
        zs              = np.random.choice(params["n_skills"], size=num_envs, p=p_z)
        episode_rewards = np.zeros(num_envs)
        episode_steps   = np.zeros(num_envs, dtype=int)
        # logq_zs buffer per env
        episode_logq    = [[] for _ in range(num_envs)]

        # Initial reset — vec_env returns (obs_batch, info_batch)
        raw_states, _ = vec_env.reset()                          # (num_envs, n_states)
        states = concat_state_latent(raw_states, zs, params["n_skills"])  # (num_envs, n_states+n_skills)

        completed = min_episode
        logger.on()

        # ---- Main loop ------------------------------------------------
        pbar = tqdm(total=params["max_n_episodes"], initial=min_episode)

        while completed < params["max_n_episodes"]:

            # 1. Choose actions for every env in one pass
            actions = np.stack([agent.choose_action(states[i]) for i in range(num_envs)])
            # shape: (num_envs, n_actions)

            # 2. Step all envs simultaneously
            raw_next, rewards, terminated, truncated, infos = vec_env.step(actions)
            dones = terminated | truncated   # (num_envs,) bool

            # 3. Gymnasium autoreset:
            #    When done[i] is True, raw_next[i] is the RESET obs (start of next ep).
            #    The true final obs is stored in infos["final_observation"][i].
            #    We must use the TRUE final obs when storing the transition.
            true_next = raw_next.copy()
            if "final_observation" in infos:
                for i in range(num_envs):
                    if dones[i]:
                        true_next[i] = infos["final_observation"][i]

            # Concatenate skill latent onto the true next states
            next_states = concat_state_latent(true_next, zs, params["n_skills"])

            # 4. Store + train once per env step
            for i in range(num_envs):
                agent.store(states[i], zs[i], dones[i], actions[i], next_states[i])
                episode_rewards[i] += rewards[i]
                episode_steps[i]   += 1

                logq_zs = agent.train()
                episode_logq[i].append(logq_zs if logq_zs is not None else last_logq_zs)

                # 5. If this env's episode ended, log and reset tracking
                if dones[i]:
                    completed += 1
                    pbar.update(1)

                    avg_logq = float(np.mean(episode_logq[i]))
                    last_logq_zs = avg_logq

                    logger.log(
                        completed,
                        episode_rewards[i],
                        zs[i],
                        avg_logq,
                        int(episode_steps[i]),
                        np.random.get_state(),
                        *agent.get_rng_states(),
                    )

                    # Reset per-env accumulators
                    episode_rewards[i] = 0
                    episode_steps[i]   = 0
                    episode_logq[i]    = []

                    # Sample a new skill for this env's next episode
                    zs[i] = np.random.choice(params["n_skills"], p=p_z)

                    if completed >= params["max_n_episodes"]:
                        break

            # 6. Build next states for continuing envs.
            #    For done envs: raw_next already holds the reset obs → use new zs[i].
            #    For live envs: use raw_next with the same zs[i].
            states = concat_state_latent(raw_next, zs, params["n_skills"])

        pbar.close()
        vec_env.close()

    # ==================================================================
    # PLAY / EVALUATE  (single env, unchanged)
    # ==================================================================
    else:
        env = gym.make(params["env_name"], render_mode="rgb_array")
        logger.load_weights()
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()
