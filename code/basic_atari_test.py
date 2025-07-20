import gymnasium as gym
import sys

# List all available environments
print("Available environments:")
from gymnasium.envs import registry
all_envs = sorted([spec.id for spec in registry.values()])
atari_envs = [env_id for env_id in all_envs if 'breakout' in env_id.lower() or 'pong' in env_id.lower()]
for env_id in atari_envs:
    print(f"  - {env_id}")

# Try to create a basic Atari environment
try:
    print("\nTrying to create Breakout-v4...")
    env = gym.make('Breakout-v4')
    print(f"Success! Action space: {env.action_space}")
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    # Try a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}, Action: {action}, Reward: {reward}")
    
    env.close()
    
except Exception as e:
    print(f"Error: {e}")

# As a fallback, try a basic environment
try:
    print("\nTrying to create CartPole-v1 as fallback...")
    env = gym.make('CartPole-v1')
    print(f"Success! Action space: {env.action_space}")
    env.close()
except Exception as e:
    print(f"Error with CartPole: {e}")

print("\nTest completed.")