#!/usr/bin/env python3
"""
Training loop for Monte-Carlo policy gradient on CartPole
"""
import numpy as np
import gymnasium as gym
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Train a policy using Monte-Carlo policy gradient.

    Args:
        env: gymnasium environment.
        nb_episodes: int, number of episodes for training.
        alpha: float, learning rate.
        gamma: float, discount factor.

    Returns:
        List of total rewards per episode.
    """
    # Initialize policy weights with small random values
    weight = np.random.randn(env.observation_space.shape[0],
                             env.action_space.n)
    # List to store total rewards per episode
    scores = []

    for episode in range(nb_episodes):
        # Reset environment at the start of each episode
        state, _ = env.reset()
        # To store rewards for the current episode
        episode_rewards = []
        grads = []  # To store gradients for each time step

        done = False
        truncated = False

        # Run an episode
        while not (done or truncated):
            # Choose action and get gradient using the policy
            action, grad = policy_gradient(state, weight)
            action = int(action)  # Ensure action is integer type

            # Store gradient
            grads.append(grad)

            # Take action in the environment
            next_state, reward, done, truncated, _ = env.step(action)

            # Store reward
            episode_rewards.append(reward)

            # Update state
            state = next_state

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)

        # Normalize returns for more stable training (optional but recommended)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Update policy weights using the gradients and returns
        for i in range(len(grads)):
            weight += alpha * returns[i] * grads[i]

        # Compute total reward for this episode
        total_reward = sum(episode_rewards)
        scores.append(total_reward)
        print(f"Episode: {episode} Score: {total_reward}")

    return scores
