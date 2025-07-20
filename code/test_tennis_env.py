# Save as test_tennis_env.py
import gymnasium as gym
import time
import os

# List available Tennis environments
from gymnasium.envs import registry
all_envs = sorted([spec.id for spec in registry.values()])
tennis_envs = [env_id for env_id in all_envs if 'tennis' in env_id.lower()]

print("Available Tennis environments:")
if tennis_envs:
    for env_id in tennis_envs:
        print(f"  - {env_id}")
else:
    print("  No Tennis environments found.")

# Try to create the Tennis environment
tennis_env_id = "ALE/Tennis-v5"  # Standard naming convention for Atari environments
print(f"\nTrying to create {tennis_env_id}...")

try:
    env = gym.make(tennis_env_id, render_mode="human")
    print(f"Successfully created environment with action space: {env.action_space}")
    
    observation, info = env.reset()
    print(f"Observation shape: {observation.shape}")
    
    print("\nRunning 100 random actions...")
    for i in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"\rStep {i+1}/100, Action: {action}, Reward: {reward}", end="")
        time.sleep(0.03)
        
        if terminated or truncated:
            observation, info = env.reset()
            print("\nEpisode ended, resetting...")
    
    env.close()
    print("\nTennis environment test successful!")
    
except Exception as e:
    print(f"Error with {tennis_env_id}: {e}")
    print("\nLet's try with a different format...")
    
    # Try alternative naming conventions
    alt_ids = ["Tennis-v5", "TennisNoFrameskip-v4", "TennisDeterministic-v4"]
    for alt_id in alt_ids:
        try:
            print(f"\nTrying {alt_id}...")
            env = gym.make(alt_id, render_mode="human")
            print(f"Success with {alt_id}!")
            env.close()
            break
        except Exception as e:
            print(f"Error with {alt_id}: {e}")

print("\nTest completed.")