# Save as list_atari_games.py
import gymnasium as gym
from gymnasium.envs import registry
import re

def list_available_atari_games():
    all_envs = sorted([spec.id for spec in registry.values()])
    
    # Extract unique Atari game names
    atari_pattern = re.compile(r'ALE/([A-Za-z0-9]+)-v\d+')
    atari_games = set()
    
    for env_id in all_envs:
        match = atari_pattern.match(env_id)
        if match:
            game_name = match.group(1)
            atari_games.add(game_name)
    
    return sorted(list(atari_games))

# Test an environment that we know works
try:
    print("Testing Breakout environment...")
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    print(f"Successfully created Breakout environment with action space: {env.action_space}")
    env.close()
except Exception as e:
    print(f"Error with Breakout: {e}")

# List all available Atari games
games = list_available_atari_games()
print(f"\nFound {len(games)} Atari games:")
for i, game in enumerate(games, 1):
    print(f"{i}. {game}")

# Check specifically for Tennis
if "Tennis" in games:
    print("\nTennis is available!")
else:
    print("\nTennis is NOT available in the installed games.")
    print("Let's list some alternative games you could use:")
    sports_games = [game for game in games if game in ["Pong", "Boxing", "IceHockey", "Football", "Skiing"]]
    for game in sports_games:
        print(f"  - {game}")

print("\nCheck complete!")