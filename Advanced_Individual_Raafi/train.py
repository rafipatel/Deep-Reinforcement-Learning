import ray.rllib.algorithms.dqn as dqn
import ray

env_name = "Breakout-v4"

config = dqn.DQNConfig()

config["gamma"] = 0.99
# config["gamma"] = tune.grid_search([0.9, 0.99])
config["train_batch_size"] = 64
config["timesteps_per_iteration"] = 1000
config["target_network_update_freq"] = 600
config["num_gpus"] = 1
config["model"]["fcnet_hiddens"] = [128, 128, 128]
config["model"]["fcnet_activation"] = "relu"
config["env"] = "BreakoutDeterministic-v4"
config["framework"] = "torch"
# config["double_q"] = True
config["n_step"] = 3
config["exploration_fraction"] = 0.1
config["exploration_final_eps"] = 0.02
config["lr"] = 0.0005
# Initialize ray
# ray.init()

# Run training

if __name__ == '__main__':

    ray.tune.run("DQN", name="Breakout_Model", stop={"timesteps_total": 100000}, checkpoint_freq=200, config=config,
                         local_dir = "/content/ray_results1221/" + env_name,
                        )

