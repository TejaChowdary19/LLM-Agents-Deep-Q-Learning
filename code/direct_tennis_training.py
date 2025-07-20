# Save as direct_tennis_training.py
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from collections import deque
import ale_py
import time
import matplotlib.pyplot as plt
from ale_py import ALEInterface
from ale_py.roms import Tennis

class DirectALEWrapper:
    """A simple wrapper for the ALE interface to make it compatible with our DQN agent"""
    def __init__(self, rom_path, render_mode="human"):
        self.ale = ALEInterface()
        
        # Set ALE options
        self.ale.setInt("random_seed", 123)
        
        if render_mode == "human":
            self.ale.setBool("display_screen", True)
        else:
            self.ale.setBool("display_screen", False)
        
        # Load the ROM
        self.ale.loadROM(rom_path)
        
        # Define action space
        self.legal_actions = self.ale.getLegalActionSet()
        self.action_space = type('DiscreteActions', (), {
            'n': len(self.legal_actions),
            'sample': lambda: random.randrange(len(self.legal_actions))
        })
        
        # Reset to get an initial observation
        self.reset()
    
    def reset(self):
        """Reset the environment and return the initial observation"""
        self.ale.reset_game()
        # Get screen dimensions
        screen_width, screen_height = self.ale.getScreenDims()
        
        # Get initial observation
        observation = self.ale.getScreenRGB()
        observation = np.array(observation, dtype=np.uint8)
        
        return observation, {}
    
    def step(self, action_idx):
        """Take a step in the environment"""
        # Convert action index to ALE action
        action = self.legal_actions[action_idx]
        
        # Take action and get reward
        reward = self.ale.act(action)
        
        # Get new observation
        observation = self.ale.getScreenRGB()
        observation = np.array(observation, dtype=np.uint8)
        
        # Check if game is over
        done = self.ale.game_over()
        
        return observation, reward, done, False, {}
    
    def close(self):
        """Close the environment"""
        pass


class DQNAgent:
    def __init__(
        self,
        state_shape,
        n_actions,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.0000009,
        memory_size=100000,
        batch_size=32,
        update_target_freq=10000
    ):
        """Initialize the Deep Q-Network agent"""
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        
        # Initialize replay memory buffer
        self.memory = deque(maxlen=memory_size)
        
        # Create main model (updated every step)
        self.model = self._build_model()
        
        # Create target model (updated every update_target_freq steps)
        self.target_model = self._build_model()
        self.update_target_network()
        
        # Training step counter
        self.training_steps = 0

    def _build_model(self):
        """Build a convolutional neural network model"""
        # Create model based on the DeepMind DQN architecture
        model = Sequential([
            # Process the image input (no need for transpose with direct ALE)
            Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_shape),
            Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            
            # Flatten and dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dense(self.n_actions, activation='linear')
        ])
        
        model.compile(loss='huber_loss', optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0))
        return model

    def update_target_network(self):
        """Update the target network with weights from the main network"""
        self.target_model.set_weights(self.model.get_weights())

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay memory buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, training=True):
        """Select an action using epsilon-greedy policy"""
        if training and np.random.rand() < self.epsilon:
            # Random action (exploration)
            return np.random.randint(self.n_actions)
        else:
            # Predict Q values and select best action (exploitation)
            state_tensor = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state_tensor, verbose=0)[0]
            return np.argmax(q_values)

    def train(self):
        """Train the model on a batch from replay memory"""
        # Ensure we have enough samples in memory
        if len(self.memory) < self.batch_size:
            return
            
        # Sample a batch of transitions from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract components from the batch
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        
        # Compute target Q values
        target_q_values = self.target_model.predict(next_states, verbose=0)
        max_target_q = np.max(target_q_values, axis=1)
        
        # Apply Bellman equation: Q(s,a) = r + γ * max_a' Q(s',a')
        targets = rewards + (1 - dones) * self.gamma * max_target_q
        
        # Get current Q values from main model
        current_q = self.model.predict(states, verbose=0)
        
        # Update only the Q values for the actions taken
        for i, action in enumerate(actions):
            current_q[i][action] = targets[i]
        
        # Train the model with updated Q values
        self.model.fit(states, current_q, epochs=1, verbose=0)
        
        # Update epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            
        # Increment training step counter
        self.training_steps += 1
        
        # Update target network if needed
        if self.training_steps % self.update_target_freq == 0:
            self.update_target_network()
            print(f"Target network updated. Epsilon: {self.epsilon:.4f}")


def preprocess_state(state):
    """Preprocess the state (observation) for the DQN"""
    # Convert to grayscale
    gray = np.mean(state, axis=2).astype(np.uint8)
    
    # Resize to 84x84
    resized = tf.image.resize(np.expand_dims(gray, axis=-1), [84, 84])
    
    # Normalize to [0, 1]
    normalized = resized / 255.0
    
    return normalized


def train_dqn_direct(rom_path, total_episodes=100, max_steps=10000):
    """Train a DQN agent directly using ALE"""
    # Create ALE environment
    env = DirectALEWrapper(rom_path, render_mode="human")
    
    # Get initial observation to determine state shape
    initial_state, _ = env.reset()
    
    # Preprocess the state
    processed_state = preprocess_state(initial_state)
    state_shape = processed_state.shape
    n_actions = env.action_space.n
    
    print(f"Original state shape: {initial_state.shape}")
    print(f"Processed state shape: {state_shape}")
    print(f"Number of actions: {n_actions}")
    
    # Create DQN agent
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.0000009,
        memory_size=100000,
        batch_size=32,
        update_target_freq=10000
    )
    
    # Training history
    training_history = {
        'episode_rewards': [],
        'epsilon_values': [],
        'avg_rewards': []
    }
    
    # Train for specified number of episodes
    for episode in range(total_episodes):
        # Reset environment and get initial state
        state, _ = env.reset()
        state = preprocess_state(state)
        episode_reward = 0
        
        # Track epsilon at the beginning of the episode
        training_history['epsilon_values'].append(agent.epsilon)
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Preprocess next state
            next_state = preprocess_state(next_state)
            
            # Store transition in replay memory
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update state and accumulated reward
            state = next_state
            episode_reward += reward
            
            # Train the agent
            agent.train()
            
            # End episode if done
            if done:
                break
        
        # Record episode reward
        training_history['episode_rewards'].append(episode_reward)
        
        # Calculate running average of episode rewards
        if len(training_history['episode_rewards']) > 100:
            avg_reward = np.mean(training_history['episode_rewards'][-100:])
        else:
            avg_reward = np.mean(training_history['episode_rewards'])
        
        training_history['avg_rewards'].append(avg_reward)
        
        # Print episode information
        print(f"Episode {episode+1}/{total_episodes} - Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    # Save final model
    agent.model.save_weights("dqn_tennis_direct_final.h5")
    
    return agent, training_history


def plot_training_history(history):
    """Plot training history metrics"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot episode rewards and running average
    episodes = range(1, len(history['episode_rewards']) + 1)
    ax1.plot(episodes, history['episode_rewards'], 'b-', alpha=0.3, label='Episode Reward')
    ax1.plot(episodes, history['avg_rewards'], 'r-', label='Running Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot epsilon decay
    ax2.plot(episodes, history['epsilon_values'], 'g-')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Exploration Rate (Epsilon) Decay')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_training_history_direct.png')
    plt.show()


if __name__ == "__main__":
    # Get Tennis ROM path
    rom_path = os.path.join(ale_py.__path__[0], "roms", "tennis.bin")
    
    if not os.path.exists(rom_path):
        print(f"Tennis ROM not found at: {rom_path}")
        exit(1)
    
    print(f"Tennis ROM found at: {rom_path}")
    
    # Training parameters
    TOTAL_EPISODES = 50  # Start with a small number for testing
    MAX_STEPS_PER_EPISODE = 10000
    
    # Start training
    print(f"Starting DQN training with direct ALE interface...")
    start_time = time.time()
    
    agent, history = train_dqn_direct(
        rom_path=rom_path,
        total_episodes=TOTAL_EPISODES,
        max_steps=MAX_STEPS_PER_EPISODE
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Plot training history
    plot_training_history(history)