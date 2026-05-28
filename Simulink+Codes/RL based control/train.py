import os
import torch
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# Imports the exact environment file you named in VS Code
from enviroment import SolarSimulinkEnv 

# 1. Verify GPU Connection
print("--------------------------------------------------")
if torch.cuda.is_available():
    print(f"SUCCESS: GPU Detected - {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: GPU not found. Training will fall back to CPU.")
print("--------------------------------------------------")

# 2. Initialize the Simulink Environment
env = SolarSimulinkEnv()

# 3. Add Exploration Noise 
# (Forces the agent to explore different duty cycles instead of guessing the same number)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 4. Build the DDPG Agent
print("Building DDPG Agent...")
model = DDPG(
    "MlpPolicy", 
    env, 
    action_noise=action_noise,
    learning_rate=1e-3,
    batch_size=128,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    device="cuda" # FORCES GPU ACCELERATION
)

# 5. Start Training Loop
print("Starting Training Loop...")
# log_interval=1 means TensorBoard updates after every single episode
model.learn(total_timesteps=50000, log_interval=1, tb_log_name="DDPG_MPPT_Run2")

# 6. Save the Brain
model.save("trained_mppt_agent")
print("Training Complete. AI Brain Saved as 'trained_mppt_agent.zip'")

env.close()