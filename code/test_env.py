import gymnasium as gym
import time

# Try different import paths for the wrappers
try:
    from gymnasium.wrappers import AtariPreprocessing, FrameStack
except ImportError:
    try:
        from gymnasium.wrappers import AtariPreprocessing
        from gymnasium.wrappers.frame_stack import FrameStack
    except ImportError:
        print("Could not import FrameStack directly. Checking available wrappers...")
        import gymnasium.wrappers
        print("Available wrappers:", dir(gymnasium.wrappers))
        raise

# Create and test the environment
env = gym.make("TennisNoFrameskip-v4", render_mode="human")
env = AtariPreprocessing(env)
env = FrameStack(env, 4)

# Reset the environment
observation, info = env.reset()

# Print information about the environment
print(f"Observation shape: {observation.shape}")
print(f"Action space: {env.action_space}")

# Run a few random actions
for _ in range(100):
    action = env.action_space.sample()  # Take random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Add a small delay to make the visualization more visible
    time.sleep(0.05)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
print("Environment test completed!")