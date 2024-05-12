#Individual Task (Raafi) - 9
#https://docs.ray.io/en/latest/rllib/rllib-algorithms.html -- to get the information of all parameters used here
# This notebook various parts are influenced by INM707 Lecture and Labs


import ray.rllib.algorithms.dqn as dqn
import ray


env_name = "Breakout-v4"

config = dqn.DQNConfig()
# config["gamma"] = tune.grid_search([0.9, 0.99])
config["gamma"] = 0.95  # Lowering the discount factor can make the agent more short-sighted and potentially learn faster.
config["train_batch_size"] = 32  # Increasing the batch size can improve the stability of the learning process.
config["timesteps_per_iteration"] = 1000  # More timesteps per iteration can provide more diverse experiences for each update.
config["target_network_update_freq"] = 800  # A less frequent update of the target network can stabilize the learning process.
config["model"]["fcnet_hiddens"] = [256, 256,256]  # A larger network might be able to learn more complex representations.
config["double_q"] = False  # Enabling Double Q-Learning can help mitigate the overestimation bias of Q-Learning.
config["n_step"] = 5  # A larger n-step return might make the learning process more stable and efficient.
config["exploration_fraction"] = 0.2  # A higher exploration fraction can encourage the agent to explore more at the beginning of training.
config["exploration_final_eps"] = 0.01  # A lower final epsilon value can make the policy less random as training progresses.
# config["lr"] = tune.grid_search([0.001, 0.005])
config["lr"] = 0.001  # A higher learning rate can make the agent learn faster, but it might also destabilize the learning process.
config["num_gpus"] = 1
config["env"] = "BreakoutDeterministic-v4"

# config["gamma"] = 0.99
# # 
# config["train_batch_size"] = 64
# config["timesteps_per_iteration"] = 1000
# config["target_network_update_freq"] = 600
# config["num_gpus"] = 1
# config["model"]["fcnet_hiddens"] = [128, 128, 128]
# config["model"]["fcnet_activation"] = "relu"
# config["env"] = "BreakoutDeterministic-v4"
# config["framework"] = "torch"
# # config["double_q"] = True
# config["n_step"] = 3
# config["exploration_fraction"] = 0.1
# config["exploration_final_eps"] = 0.02
# config["lr"] = 0.0005
# # Initialize ray
# ray.init()

# Run training

if __name__ == '__main__':

    ray.tune.run("DQN", name="Breakout_Model", stop={"timesteps_total": 100000}, checkpoint_freq=20, config=config,
                         local_dir = "/content/ray_results_model_dqn/" + env_name,
                        )

