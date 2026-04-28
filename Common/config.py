import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--env_name", default="HalfCheetah-v5", type=str,
                        help="Name of the environment. Try: HalfCheetah-v5, Humanoid-v5, Ant-v4, Hopper-v4")
    parser.add_argument("--interval", default=100, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by episodes.")
    parser.add_argument("--do_train", action="store_true",
                        help="The flag determines whether to train the agent or play with it.")
    parser.add_argument("--train_from_scratch", action="store_true",
                        help="The flag determines whether to train from scratch or continue previous tries.")
    parser.add_argument("--mem_size", default=int(1e+6), type=int, help="The memory size.")
    parser.add_argument("--n_skills", default=50, type=int, help="The number of skills to learn.")
    parser.add_argument("--reward_scale", default=1, type=float, help="The reward scaling factor introduced in SAC.")
    parser.add_argument("--seed", default=123, type=int,
                        help="The randomness' seed for torch, numpy, random & gym[env].")

    # NEW: number of parallel environments.
    # Each env runs on its own CPU thread stepping MuJoCo independently.
    # On Kaggle (2 vCPUs), 4 is a safe sweet spot — beyond 4 you get diminishing
    # returns because Kaggle only has 2 physical cores.
    parser.add_argument("--num_envs", default=4, type=int,
                        help="Number of parallel envs for vectorized training (default: 4).")

    parser_params = parser.parse_args()

    #  Parameters based on the DIAYN and SAC papers.
    default_params = {"lr": 3e-4,
                      "batch_size": 256,
                      "max_n_episodes": 10000,
                      "max_episode_len": 1000,
                      "gamma": 0.99,
                      "alpha": 0.1,
                      "tau": 0.005,
                      "n_hiddens": 512
                      }

    total_params = {**vars(parser_params), **default_params}
    return total_params
