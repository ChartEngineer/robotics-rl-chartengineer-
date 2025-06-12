"""
Deep Q-Network Agent for Robotic Arm Control

This module implements a DQN agent for learning robotic arm control tasks.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import matplotlib.pyplot as plt

class DQNAgent:
    """
    Deep Q-Network agent for robotic arm control.
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        """
        Initialize DQN agent.
        
        Args:
            state_size (int): Size of state space
            action_size (int): Size of action space
            learning_rate (float): Learning rate for neural network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        self.batch_size = 32
        self.memory_size = 10000
        self.target_update_freq = 100
        
        # Experience replay memory
        self.memory = deque(maxlen=self.memory_size)
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'epsilon_values': []
        }
        
    def _build_model(self):
        """
        Build the neural network model.
        
        Returns:
            keras.Model: DQN model
        """
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        
        Args:
            state (np.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.array): Next state
            done (bool): Whether episode ended
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state (np.array): Current state
            training (bool): Whether in training mode
            
        Returns:
            int: Selected action
        """
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """
        Train the model on a batch of experiences.
        
        Returns:
            float: Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q values
        target_q_values = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        history = self.q_network.fit(
            states, target_q_values,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0
        )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def update_target_network(self):
        """
        Update target network weights.
        """
        self.target_network.set_weights(self.q_network.get_weights())
    
    def train(self, env, episodes=1000, max_steps_per_episode=200):
        """
        Train the DQN agent.
        
        Args:
            env: Environment to train on
            episodes (int): Number of episodes to train
            max_steps_per_episode (int): Maximum steps per episode
            
        Returns:
            dict: Training history
        """
        print(f"Starting DQN training for {episodes} episodes...")
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Choose action
                action = self.act(state, training=True)
                
                # Take action
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store experience
                self.remember(state, action, reward, next_state, done or truncated)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Train the model
                if len(self.memory) > self.batch_size:
                    loss = self.replay()
                    self.training_history['losses'].append(loss)
                
                if done or truncated:
                    break
            
            # Update target network periodically
            if episode % self.target_update_freq == 0:
                self.update_target_network()
            
            # Record training metrics
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['episode_lengths'].append(steps)
            self.training_history['epsilon_values'].append(self.epsilon)
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                success_rate = np.mean([info.get('success', False) for info in [{}] * 100])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, Success Rate: {success_rate:.2f}")
        
        print("Training completed!")
        return self.training_history
    
    def evaluate(self, env, episodes=100, render=False):
        """
        Evaluate the trained agent.
        
        Args:
            env: Environment to evaluate on
            episodes (int): Number of evaluation episodes
            render (bool): Whether to render episodes
            
        Returns:
            dict: Evaluation results
        """
        print(f"Evaluating agent for {episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                if render:
                    env.render()
                
                # Choose action (no exploration)
                action = self.act(state, training=False)
                
                # Take action
                state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if done or truncated:
                    if info.get('success', False):
                        success_count += 1
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        results = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'success_rate': success_count / episodes,
            'episode_rewards': episode_rewards
        }
        
        print(f"Evaluation Results:")
        print(f"  Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Average Episode Length: {results['avg_length']:.1f}")
        print(f"  Success Rate: {results['success_rate']:.2%}")
        
        return results
    
    def plot_training_history(self):
        """
        Plot training history.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.training_history['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Training loss
        if self.training_history['losses']:
            axes[1, 0].plot(self.training_history['losses'])
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
        
        # Epsilon decay
        axes[1, 1].plot(self.training_history['epsilon_values'])
        axes[1, 1].set_title('Epsilon Decay')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        self.q_network.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.q_network = keras.models.load_model(filepath)
        self.target_network = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

class DoubleDQNAgent(DQNAgent):
    """
    Double DQN agent with improved stability.
    """
    
    def replay(self):
        """
        Train using Double DQN algorithm.
        
        Returns:
            float: Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Double DQN: use main network to select actions, target network to evaluate
        next_q_values_main = self.q_network.predict(next_states, verbose=0)
        next_q_values_target = self.target_network.predict(next_states, verbose=0)
        
        # Update Q values
        target_q_values = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                # Select action with main network, evaluate with target network
                best_action = np.argmax(next_q_values_main[i])
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * next_q_values_target[i][best_action]
        
        # Train the model
        history = self.q_network.fit(
            states, target_q_values,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0
        )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]

