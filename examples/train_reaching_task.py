"""
Example: Training a robotic arm to reach targets

This example demonstrates how to train a DQN agent to control
a robotic arm for reaching tasks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.environment import SimpleRobotArmEnv  # Use simple env for demo
from src.dqn_agent import DQNAgent

def main():
    """
    Main function for training robotic arm reaching task.
    """
    print("Robotic Arm Reaching Task Training")
    print("="*50)
    
    # Create environment
    env = SimpleRobotArmEnv(num_joints=3)
    
    print(f"Environment created:")
    print(f"  State space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")
    
    # Create DQN agent
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        learning_rate=0.001
    )
    
    print(f"\nDQN Agent created:")
    print(f"  State size: {agent.state_size}")
    print(f"  Action size: {agent.action_size}")
    print(f"  Network architecture: {agent.q_network.summary()}")
    
    # Train the agent
    print(f"\nStarting training...")
    training_history = agent.train(env, episodes=500, max_steps_per_episode=200)
    
    # Plot training results
    print(f"\nPlotting training results...")
    agent.plot_training_history()
    
    # Evaluate the trained agent
    print(f"\nEvaluating trained agent...")
    eval_results = agent.evaluate(env, episodes=100, render=False)
    
    # Save the trained model
    model_path = 'models/reaching_task_dqn.h5'
    os.makedirs('models', exist_ok=True)
    agent.save_model(model_path)
    
    # Demonstrate learned behavior
    print(f"\nDemonstrating learned behavior...")
    demonstrate_agent(env, agent, episodes=5)
    
    print(f"\nTraining completed successfully!")
    print(f"Final success rate: {eval_results['success_rate']:.2%}")

def demonstrate_agent(env, agent, episodes=5):
    """
    Demonstrate the trained agent's behavior.
    
    Args:
        env: Environment
        agent: Trained agent
        episodes (int): Number of episodes to demonstrate
    """
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Initial state: {state}")
        print(f"  Target position: {state[-2:]}")
        
        while steps < 200:
            action = agent.act(state, training=False)
            next_state, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done or truncated:
                break
        
        print(f"  Final state: {state}")
        print(f"  End effector position: {state[-4:-2]}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps taken: {steps}")
        print(f"  Success: {info.get('success', False)}")
        print(f"  Final distance: {info.get('distance', 0):.3f}")

if __name__ == "__main__":
    main()

