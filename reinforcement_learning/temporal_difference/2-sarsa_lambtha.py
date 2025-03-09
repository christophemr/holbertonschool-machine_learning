#!/usr/bin/env python3
"""
SARSA(λ) algorithm with eligibility traces for RL
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Determines the next action using the epsilon-greedy policy.

    Parameters:
        Q (numpy.ndarray): The Q-table, where each entry Q[s, a] represents
            the expected reward for state `s` and action `a`.
        state (int): The current state.
        epsilon (float): The epsilon value for the epsilon-greedy policy.

    Returns:
        int: The index of the action to take next.
    """
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state, :])  # Exploitation
    else:
        return np.random.randint(0, Q.shape[1])  # Exploration


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm (with eligibility traces) to estimate
    a Q-table.

    Parameters:
        env (gym.Env): The FrozenLake environment instance.
        Q (numpy.ndarray): The given Q-table.
        lambtha (float): The eligibility trace factor.
        episodes (int, optional): The total number of episodes to train over.
        max_steps (int, optional): The maximum number of steps per episode.
        alpha (float, optional): The learning rate.
        gamma (float, optional): The discount rate.
        epsilon (float, optional): The initial threshold for epsilon-greedy.
        min_epsilon (float, optional): The minimum value for epsilon.
        epsilon_decay (float, optional): The decay rate for epsilon per episode

    Returns:
        numpy.ndarray: The updated Q-table after training.
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        # Reset environment and choose first action
        state, _ = env.reset()  # Correction pour Gym v26+
        action = epsilon_greedy(Q, state, epsilon)

        # Initialize eligibility traces
        eligibility_traces = np.zeros_like(Q)

        for _ in range(max_steps):
            # Take action and observe results
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon)

            # Compute TD Error (δ)
            delta = (reward + gamma
                     * Q[new_state, new_action] - Q[state, action])

            # Update eligibility trace for current state-action pair
            eligibility_traces[state, action] += 1  # Accumulating traces

            # Update Q-table using eligibility traces
            Q += alpha * delta * eligibility_traces

            # Decay eligibility traces
            eligibility_traces *= gamma * lambtha

            # Transition to new state/action
            if terminated or truncated:
                break
            state = new_state
            action = new_action

        # Decay epsilon
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon)
                   * np.exp(-epsilon_decay * episode))

    return Q
