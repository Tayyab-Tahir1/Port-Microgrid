import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

# Import your custom EnergyEnv from models/environment.py
from models.environment import EnergyEnv

data_lines = 8760
episodes_required = 200
timesteps_cal = data_lines * episodes_required

# -------------------------------
# Gymnasium Wrapper for EnergyEnv
# -------------------------------
class EnergyEnvGym(gym.Env):
    """
    Wraps your custom EnergyEnv into a Gymnasium environment.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, data):
        super(EnergyEnvGym, self).__init__()
        self.env = EnergyEnv(data)
        # Updated: Environment state is now 9-dimensional
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
        # Updated: Expanded action space: 8 discrete actions (0-7)
        self.action_space = spaces.Discrete(8)
        
    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return np.array(obs, dtype=np.float32), {}
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return np.array(state, dtype=np.float32), reward, done, False, info
    
    def render(self, mode="human"):
        soc = self.env.soc / self.env.battery_capacity * 100
        print(f"Step: {self.env.current_step}, Battery SoC: {soc:.2f}%")

# -------------------------------
# TensorBoard Logging Callback
# -------------------------------
class TensorBoardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorBoardLoggingCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        # Log reward from the current step.
        reward = self.locals.get("rewards", [0])[0]
        self.logger.record("custom/reward", reward)
        # Log loss if available.
        loss = self.locals.get("loss", None)
        if loss is not None:
            self.logger.record("custom/loss", loss)
        return True

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_data(data):
    """
    Rename dataset columns so they match what your EnergyEnv expects.
    For example, if your dataset uses 'Tou Tariff' and 'H2 Tariff', they are renamed.
    """
    rename_mapping = {
        'Tou Tariff': 'Tou_Tariff',
        'H2 Tariff': 'H2_Tariff'
    }
    data.rename(columns=rename_mapping, inplace=True)
    return data

# -------------------------------
# Main Function (Fine-Tuned Version)
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and Test Fine-Tuned DQN on Energy Management Environment")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], default="both",
                        help="Mode to run: train, test, or both")
    parser.add_argument("--timesteps", type=int, default=timesteps_cal,
                        help="Total timesteps for training")
    parser.add_argument("--test_episodes", type=int, default=10,
                        help="Number of episodes for testing")
    args = parser.parse_args()

    # Load dataset from training_data.csv
    dataset_path = 'dataset.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    data = pd.read_csv(dataset_path)
    data = preprocess_data(data)
    
    # Create Gymnasium environment wrapper around your EnergyEnv
    env = EnergyEnvGym(data)
    check_env(env, warn=True)
    
    # Define model save path
    model_path = "dqn_energy_model_finetuned"
    
    # Define custom policy parameters: Two hidden layers with 128 neurons each.
    policy_kwargs = dict(net_arch=[128, 128])
    
    if args.mode in ["train", "both"]:
        print("Training mode selected.")
        # Fine-tuning hyperparameters
        model = DQN(
            "MlpPolicy", 
            env, 
            learning_rate=1e-4,
            buffer_size=100000,
            exploration_fraction=0.8,
            exploration_final_eps=0.1,
            gamma=0.99,
            batch_size=64,
            policy_kwargs=policy_kwargs,
            verbose=1, 
            tensorboard_log="./tb_logs_finetuned/"
        )
        # Train the model with the TensorBoard callback for logging loss and reward.
        model.learn(total_timesteps=args.timesteps, callback=TensorBoardLoggingCallback())
        model.save(model_path)
        print(f"Model saved as {model_path}.zip")
    else:
        if os.path.exists(model_path + ".zip"):
            model = DQN.load(model_path, env=env)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}.zip")
    
    if args.mode in ["test", "both"]:
        print("Testing mode selected.")
        for ep in range(args.test_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            print(f"\nStarting test episode {ep+1}:")
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward
                env.render()  # Render current step info
                if done or trunc:
                    break
            print(f"Episode {ep+1} finished with total reward: {total_reward:.2f}")
    
if __name__ == "__main__":
    main()
