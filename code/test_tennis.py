import gymnasium as gym
import os
import time

# Set the ROM directory environment variable
os.environ["ATARI_ROM_DIR"] = os.path.join(os.getcwd(), "roms")

# Try different environment IDs for Tennis
env_ids = [
    "ALE/Tennis-v5",
    "Tennis-v0",
    "Tennis-v4",
    "TennisNoFrameskip-v4"
]

# Try each environment ID
for env_id in env_ids:
    try:
        print(f"\nTrying environment: {env_id}")
        env = gym.make(env_id, render_mode="human")
        print(f"Successfully created environment with action space: {env.action_space}")
        
        observation, info = env.reset()
        print(f"Observation shape: {observation.shape}")
        
        for i in range(50):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i}, Action: {action}, Reward: {reward}")
            time.sleep(0.05)
            
            if terminated or truncated:
                observation, info = env.reset()
                
        env.close()
        print(f"Successfully tested {env_id}")
        break  # Stop after finding a working environment
        
    except Exception as e:
        print(f"Error with {env_id}: {e}")

print("\nTest completed.")