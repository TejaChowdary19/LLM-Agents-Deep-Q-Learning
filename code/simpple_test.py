import gymnasium as gym
import time

# Create a simple environment first to test if Gymnasium works
try:
    # Try to create a simple environment first
    print("Testing with a simple CartPole environment...")
    simple_env = gym.make("CartPole-v1", render_mode="human")
    simple_env.reset()
    for _ in range(100):
        action = simple_env.action_space.sample()
        observation, reward, terminated, truncated, info = simple_env.step(action)
        time.sleep(0.05)
        if terminated or truncated:
            simple_env.reset()
    simple_env.close()
    print("CartPole test completed successfully!")
except Exception as e:
    print(f"Error with CartPole: {e}")

# Now try with Atari Tennis
try:
    print("\nTesting Atari Tennis environment...")
    env = gym.make("TennisNoFrameskip-v4", render_mode="human")
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()  # Take random action
        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.05)
        if terminated or truncated:
            env.reset()
    env.close()
    print("Tennis test completed successfully!")
except Exception as e:
    print(f"Error with Tennis: {e}")