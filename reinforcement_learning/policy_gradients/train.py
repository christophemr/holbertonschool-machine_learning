#!/usr/bin/env python3
"""
Training loop for Monte-Carlo policy gradient on CartPole.
"""

import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.01, gamma=0.99):
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
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    weights = np.random.randn(n_states, n_actions) * 1e-2
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        done = False
        rewards = []
        grads = []

        while not done:
            action, grad = policy_gradient(state, weights)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            grads.append(grad)
            rewards.append(reward)
            state = next_state

        # Compute discounted returns
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)  # insert to keep correct order

        returns = np.array(returns)

        # Normalize returns
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Update weights
        for grad, Gt in zip(grads, returns):
            weights -= alpha * grad * Gt

        score = sum(rewards)
        scores.append(score)
        print(f"Episode: {episode} Score: {score}")

    return scores
