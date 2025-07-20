import gymnasium as gym
from gymnasium.envs.registration import registry

# List all registered environments that contain 'tennis' (case insensitive)
tennis_envs = [env_id for env_id in registry.keys() if 'tennis' in env_id.lower()]
print("Available Tennis environments:")
for env_id in tennis_envs:
    print(f"  - {env_id}")

# List some Atari environments as a fallback
print("\nSome available Atari environments:")
atari_envs = [env_id for env_id in registry.keys() if 'atari' in env_id.lower() or 'breakout' in env_id.lower()][:10]
for env_id in atari_envs:
    print(f"  - {env_id}")

# Try to create a random Atari environment as a test
if atari_envs:
    try:
        print(f"\nTesting environment: {atari_envs[0]}")
        env = gym.make(atari_envs[0], render_mode="human")
        env.reset()
        for _ in range(10):
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset()
        env.close()
        print(f"Successfully tested {atari_envs[0]}")
    except Exception as e:
        print(f"Error testing {atari_envs[0]}: {e}")