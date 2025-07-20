# Create a new file called simple_dqn_tennis.py with this content
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from collections import deque
import gymnasium as gym
import time
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Create a simpler version to test with
ENV_NAME = "Tennis-v0"  # Try the direct name format
TOTAL_EPISODES = 10     # Small number for testing
RENDER_MODE = "human"   # Enable rendering

print(f"Attempting to create environment: {ENV_NAME}")
try:
    env = gym.make(ENV_NAME, render_mode=RENDER_MODE)
    print(f"Successfully created environment!")
    print(f"Action space: {env.action_space}")
    
    # Test the environment
    state, _ = env.reset()
    print(f"Observation shape: {state.shape}")
    
    # Run a few random actions
    for i in range(100):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"\rStep {i+1}/100, Action: {action}, Reward: {reward}", end="")
        time.sleep(0.03)
        
        if terminated or truncated:
            state, _ = env.reset()
            print("\nEpisode ended, resetting...")
    
    env.close()
    print("\nEnvironment test successful!")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nAttempting to list available environments containing 'tennis'...")
    
    try:
        # List all registered environments that contain 'tennis' (case insensitive)
        from gymnasium.envs.registration import registry
        tennis_envs = [env_id for env_id in registry.keys() if 'tennis' in env_id.lower()]
        print("Available Tennis environments:")
        if tennis_envs:
            for env_id in tennis_envs:
                print(f"  - {env_id}")
            
            # Try the first available Tennis environment
            test_env_id = tennis_envs[0]
            print(f"\nTrying to use: {test_env_id}")
            test_env = gym.make(test_env_id, render_mode=RENDER_MODE)
            test_env.reset()
            test_env.close()
            print(f"Success with {test_env_id}! Use this ID in your DQN implementation.")
        else:
            print("No Tennis environments found.")
            
            # Try a few other common Atari games
            alt_games = ["Pong-v0", "Breakout-v0", "SpaceInvaders-v0"]
            for game in alt_games:
                try:
                    print(f"\nTrying alternative game: {game}")
                    alt_env = gym.make(game, render_mode=RENDER_MODE)
                    alt_env.reset()
                    alt_env.close()
                    print(f"Success with {game}! You can use this as an alternative.")
                    break
                except Exception as alt_e:
                    print(f"Error with {game}: {alt_e}")
    
    except Exception as list_e:
        print(f"Error listing environments: {list_e}")

print("\nTest complete!")