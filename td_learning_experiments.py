"""
TD Learning Algorithms Comparison
CS5100 FAI Capstone - Milestone 2

This code compares three Temporal Difference (TD) learning algorithms:
1. Q-learning (Watkins & Dayan, 1992) - Off-policy TD control
2. SARSA (Rummery & Niranjan, 1994) - On-policy TD control
3. Double Q-learning (Hasselt, 2010) - Reduces overestimation bias

Key Differences:
- Q-learning: Uses max Q(s',a') for update (learns optimal policy)
- SARSA: Uses actual next action's Q(s',a') for update (learns actual policy)
- Double Q-learning: Uses two Q-tables to reduce overestimation

Environments tested:
- GridWorld: Simple deterministic grid navigation
- CliffWalking: Grid with dangerous cliff (tests risk handling)
- FrozenLake: Stochastic environment (agent can slip on ice)

"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import random
import gymnasium as gym  # Gymnasium: Standard library for RL environments


# ENVIRONMENT CLASSES

class GridWorld:
    """
    Custom GridWorld Environment (same as in qlearning_gridworld.py).
    
    A simple grid where agent navigates from start (0,0) to goal (size-1, size-1).
    
    Properties:
    - Deterministic: Actions always have the expected result
    - No obstacles: Agent can move freely
    - Small state space: Easy to learn optimal policy
    """
    
    def __init__(self, size: int = 5):
        """Initialize GridWorld with given size."""
        self.size = size
        self.n_states = size * size      # Total states: 25 for 5x5
        self.n_actions = 4               # Actions: up, down, left, right
        self.state = None
        self.goal = (size - 1, size - 1)
        self.action_names = ['up', 'down', 'left', 'right']
        
    def reset(self) -> int:
        """Reset to start state and return state index."""
        self.state = (0, 0)
        return self._state_to_idx(self.state)
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """
        Take action and return (next_state, reward, terminated, truncated, info).
        
        This follows Gymnasium's API format for compatibility.
        """
        row, col = self.state
        
        # Apply action with boundary checking
        if action == 0:    # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.size - 1, col + 1)
        
        self.state = (row, col)
        done = (self.state == self.goal)
        reward = 1.0 if done else -0.01
        
        # Return format: (state, reward, terminated, truncated, info)
        return self._state_to_idx(self.state), reward, done, False, {}
    
    def _state_to_idx(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) to single index."""
        return state[0] * self.size + state[1]
    
    def _idx_to_state(self, idx: int) -> Tuple[int, int]:
        """Convert single index to (row, col)."""
        return (idx // self.size, idx % self.size)


class GymnasiumWrapper:
    """
    Wrapper for Gymnasium environments to provide consistent interface.
    
    Gymnasium provides standard RL environments:
    - CliffWalking-v1: Grid with cliff, tests risk avoidance
    - FrozenLake-v1: Slippery ice, tests handling uncertainty
    
    This wrapper extracts n_states and n_actions from the environment.
    """
    
    def __init__(self, env_name: str):
        """
        Initialize wrapper with a Gymnasium environment.
        
        Args:
            env_name: Name of the Gymnasium environment (e.g., 'CliffWalking-v1')
        """
        self.env = gym.make(env_name)
        # Get state space size (number of discrete states)
        self.n_states = self.env.observation_space.n
        # Get action space size (number of discrete actions)
        self.n_actions = self.env.action_space.n
        
    def reset(self) -> int:
        """Reset environment and return initial state."""
        state, _ = self.env.reset()
        return state
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """Take action and return result."""
        return self.env.step(action)


# AGENT CLASSES - Three Different TD Learning Algorithms

class QLearningAgent:
    """
    Q-learning Agent: Off-policy TD Control (Watkins & Dayan, 1992)
    
    Key characteristic: Uses MAX Q(s',a') in update rule.
    This means it learns the OPTIMAL policy regardless of what actions
    it actually takes during exploration.
    
    Update rule: Q(s,a) <- Q(s,a) + alpha * [r + gamma * MAX Q(s',a') - Q(s,a)]
                                                       ^^^
                                            Uses maximum over all actions
    
    Advantage: Learns optimal policy
    Disadvantage: Can overestimate Q-values, may take risky actions during learning
    """
    
    def __init__(self, n_states: int, n_actions: int, 
                 alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        """
        Initialize Q-learning agent.
        
        Args:
            n_states: Number of states in environment
            n_actions: Number of possible actions
            alpha: Learning rate (how fast to learn)
            gamma: Discount factor (importance of future rewards)
            epsilon: Exploration rate (probability of random action)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        self.name = "Q-learning"
    
    def choose_action(self, state: int) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        With probability epsilon: random action (exploration)
        With probability 1-epsilon: best action (exploitation)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Random action
        return int(np.argmax(self.q_table[state]))        # Best action
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, next_action: int, done: bool):
        """
        Update Q-value using Q-learning rule.
        
        Q-learning uses MAX Q(s',a') - this is what makes it "off-policy".
        It doesn't care what action will actually be taken next.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Not used in Q-learning (only used by SARSA)
            done: Whether episode ended
        """
        if done:
            # No future reward if episode ended
            target = reward
        else:
            # Q-learning: use MAX over next state's Q-values
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Update Q-value
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])
    
    def get_policy(self) -> np.ndarray:
        """Return best action for each state."""
        return np.argmax(self.q_table, axis=1)


class SARSAAgent:
    """
    SARSA Agent: On-policy TD Control (Rummery & Niranjan, 1994)
    
    SARSA = State-Action-Reward-State-Action
    
    Key characteristic: Uses ACTUAL next action Q(s',a') in update rule.
    This means it learns the policy it's ACTUALLY following, including
    the exploration actions.
    
    Update rule: Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
                                                       ^^^^
                                            Uses actual next action, not max
    
    Advantage: More conservative, considers exploration risk
    Disadvantage: Doesn't learn optimal policy directly
    
    In CliffWalking:
    - Q-learning learns the path close to cliff (optimal but risky)
    - SARSA learns a safer path away from cliff (considers exploration risk)
    """
    
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        """Initialize SARSA agent (same parameters as Q-learning)."""
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))
        self.name = "SARSA"
    
    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy (same as Q-learning)."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state]))
    
    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int, done: bool):
        """
        Update Q-value using SARSA rule.
        
        SARSA uses the ACTUAL next action's Q-value.
        This is what makes it "on-policy" - it learns about the policy
        it's actually following.
        
        KEY DIFFERENCE FROM Q-LEARNING:
        - Q-learning: target = r + gamma * MAX Q(s',a')
        - SARSA:      target = r + gamma * Q(s', next_action)
        """
        if done:
            target = reward
        else:
            # SARSA: use Q-value of ACTUAL next action (not max!)
            target = reward + self.gamma * self.q_table[next_state, next_action]
        
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])
    
    def get_policy(self) -> np.ndarray:
        """Return best action for each state."""
        return np.argmax(self.q_table, axis=1)


class DoubleQLearningAgent:
    """
    Double Q-learning Agent (Hasselt, 2010)
    
    Problem with regular Q-learning:
    Q-learning uses max Q(s',a'), which tends to OVERESTIMATE Q-values.
    This is because max of noisy estimates is biased upward.
    
    Solution: Use TWO Q-tables (Q1 and Q2)
    - With 50% probability: use Q1 to select action, Q2 to evaluate
    - With 50% probability: use Q2 to select action, Q1 to evaluate
    
    This "decouples" action selection from action evaluation,
    reducing overestimation bias.
    
    Update (50% probability each):
    - Q1(s,a) <- Q1(s,a) + alpha * [r + gamma * Q2(s', argmax_a' Q1(s',a')) - Q1(s,a)]
    - Q2(s,a) <- Q2(s,a) + alpha * [r + gamma * Q1(s', argmax_a' Q2(s',a')) - Q2(s,a)]
    """
    
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        """Initialize Double Q-learning with TWO Q-tables."""
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # TWO Q-tables instead of one
        self.q_table1 = np.zeros((n_states, n_actions))
        self.q_table2 = np.zeros((n_states, n_actions))
        self.name = "Double Q-learning"
    
    def choose_action(self, state: int) -> int:
        """
        Choose action using epsilon-greedy on COMBINED Q-values.
        We sum both Q-tables for action selection.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        # Use sum of both Q-tables for action selection
        combined_q = self.q_table1[state] + self.q_table2[state]
        return int(np.argmax(combined_q))
    
    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int, done: bool):
        """
        Update Q-value using Double Q-learning rule.
        
        Randomly choose which Q-table to update (50/50).
        Use one table to SELECT the best action, and the OTHER to EVALUATE it.
        This decoupling reduces overestimation.
        """
        # Randomly choose which Q-table to update
        if random.random() < 0.5:
            # Update Q1 using Q2 for evaluation
            if done:
                target = reward
            else:
                # Use Q1 to select best action
                best_action = np.argmax(self.q_table1[next_state])
                # Use Q2 to evaluate that action (this is the key!)
                target = reward + self.gamma * self.q_table2[next_state, best_action]
            self.q_table1[state, action] += self.alpha * (target - self.q_table1[state, action])
        else:
            # Update Q2 using Q1 for evaluation
            if done:
                target = reward
            else:
                # Use Q2 to select best action
                best_action = np.argmax(self.q_table2[next_state])
                # Use Q1 to evaluate that action
                target = reward + self.gamma * self.q_table1[next_state, best_action]
            self.q_table2[state, action] += self.alpha * (target - self.q_table2[state, action])
    
    @property
    def q_table(self):
        """Return average of both Q-tables for policy extraction."""
        return (self.q_table1 + self.q_table2) / 2
    
    def get_policy(self) -> np.ndarray:
        """Return best action for each state."""
        return np.argmax(self.q_table, axis=1)


# TRAINING FUNCTION

def train_agent(env, agent, n_episodes: int = 500, max_steps: int = 200) -> List[float]:
    """
    Train an agent using the standard RL loop.
    
    The training loop:
    1. Reset environment
    2. Choose initial action
    3. Repeat until done:
       a. Take action, observe reward and next state
       b. Choose next action
       c. Update Q-values
       d. Move to next state/action
    
    Args:
        env: The environment to train in
        agent: The learning agent (Q-learning, SARSA, or Double Q)
        n_episodes: Number of episodes to train = 500
        max_steps: Maximum steps per episode (prevent infinite loops)
        
    Returns:
        List of total rewards per episode
    """
    episode_rewards = []
    
    for episode in range(n_episodes):
        # Reset environment and choose initial action
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        
        for step in range(max_steps):
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Choose next action (needed for SARSA)
            next_action = agent.choose_action(next_state)
            
            # Update agent's Q-values
            # Note: Q-learning ignores next_action, SARSA uses it
            agent.update(state, action, reward, next_state, next_action, done)
            
            # Accumulate reward
            total_reward += reward
            
            # Move to next state and action
            state = next_state
            action = next_action
            
            if done:
                break
        
        episode_rewards.append(total_reward)
    
    return episode_rewards



# EXPERIMENT RUNNER

def run_experiment(env_class, env_kwargs: dict, agent_classes: list, 
                   agent_kwargs: dict, n_episodes: int = 500, n_runs: int = 5) -> Dict:
    """
    Run experiments with multiple agents on an environment.
    
    For statistical significance, we run each agent multiple times
    and average the results.
    
    Args:
        env_class: Environment class to use
        env_kwargs: Arguments for environment constructor
        agent_classes: List of agent classes to compare
        agent_kwargs: Arguments for agent constructors
        n_episodes: Episodes per run = 500
        n_runs: Number of runs to average over = 5
        
    Returns:
        Dictionary with results for each agent
    """
    results = {}
    
    for AgentClass in agent_classes:
        print(f"  Training {AgentClass.__name__}...")
        all_rewards = []
        
        for run in range(n_runs):
            # Create fresh environment and agent for each run
            if env_kwargs.get('gym_env'):
                # Use Gymnasium environment
                env = GymnasiumWrapper(env_kwargs['gym_env'])
            else:
                # Use custom environment
                env = env_class(**env_kwargs)
            
            # Create agent
            agent = AgentClass(
                n_states=env.n_states,
                n_actions=env.n_actions,
                **agent_kwargs
            )
            
            # Train and record rewards
            rewards = train_agent(env, agent, n_episodes)
            all_rewards.append(rewards)
        
        # Store results with mean and standard deviation
        results[AgentClass.__name__] = {
            'rewards': np.array(all_rewards),
            'mean': np.mean(all_rewards, axis=0),
            'std': np.std(all_rewards, axis=0) # 对同一 episode 在多个 runs 上求标准差
        }
    
    return results


# VISUALIZATION FUNCTIONS


def plot_learning_curves(results: Dict, title: str, save_path: str = None):
    """
    Plot learning curves for all agents with confidence intervals.
    
    Shows:
    - Mean reward over episodes (solid line)
    - Standard deviation (shaded area)
    
    This helps compare:
    - How fast each algorithm learns
    - Final performance
    - Stability across runs
    """
    plt.figure(figsize=(10, 6))
    
    # Colors for each algorithm
    colors = {'QLearningAgent': 'blue', 'SARSAAgent': 'green', 'DoubleQLearningAgent': 'red'}
    labels = {'QLearningAgent': 'Q-learning', 'SARSAAgent': 'SARSA', 'DoubleQLearningAgent': 'Double Q-learning'}
    
    for agent_name, data in results.items():
        mean = data['mean']
        std = data['std']
        episodes = np.arange(len(mean))
        
        # Smooth the curves with moving average for better visualization
        window = 20
        if len(mean) >= window:
            mean_smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
            std_smooth = np.convolve(std, np.ones(window)/window, mode='valid')
            episodes_smooth = episodes[window-1:]
        else:
            mean_smooth, std_smooth, episodes_smooth = mean, std, episodes
        
        color = colors.get(agent_name, 'gray')
        label = labels.get(agent_name, agent_name)
        
        # Plot mean with confidence interval
        plt.plot(episodes_smooth, mean_smooth, color=color, label=label, linewidth=2)
        plt.fill_between(episodes_smooth, 
                        mean_smooth - std_smooth, 
                        mean_smooth + std_smooth,
                        color=color, alpha=0.2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_comparison_summary(all_results: Dict, save_path: str = None):
    """
    Plot bar chart comparing final performance across environments.
    
    Shows average reward over last 50 episodes for each algorithm
    in each environment.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    envs = list(all_results.keys())
    agents = ['QLearningAgent', 'SARSAAgent', 'DoubleQLearningAgent']
    agent_labels = ['Q-learning', 'SARSA', 'Double Q-learning']
    
    x = np.arange(len(envs))
    width = 0.25
    
    colors = ['blue', 'green', 'red']
    
    for i, (agent, label) in enumerate(zip(agents, agent_labels)):
        means = []
        stds = []
        for env_name in envs:
            if agent in all_results[env_name]:
                # Average of last 50 episodes
                final_rewards = all_results[env_name][agent]['rewards'][:, -50:]
                means.append(np.mean(final_rewards))
                stds.append(np.std(np.mean(final_rewards, axis=1)))
            else:
                means.append(0)
                stds.append(0)
        
        ax.bar(x + i*width, means, width, label=label, color=colors[i], alpha=0.8, yerr=stds)
    
    ax.set_xlabel('Environment', fontsize=12)
    ax.set_ylabel('Average Reward (last 50 episodes)', fontsize=12)
    ax.set_title('Algorithm Comparison Across Environments', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(envs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()



# MAIN PROGRAM

if __name__ == "__main__":
    """
    Main program - compares Q-learning, SARSA, and Double Q-learning
    on three different environments.
    
    Experiments:
    1. GridWorld: Simple, deterministic (baseline test)
    2. CliffWalking: Has dangerous cliff (tests risk handling)
    3. FrozenLake: Stochastic transitions (tests uncertainty handling)
    """
    
    print("TD Learning Algorithms Comparison - CS5100 Capstone")
    
    # Configuration
    agent_classes = [QLearningAgent, SARSAAgent, DoubleQLearningAgent]
    agent_kwargs = {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 0.1}
    n_episodes = 500
    n_runs = 5  # Run each experiment 5 times for statistics
    
    all_results = {}
    
    # EXPERIMENT 1: GridWorld (Simple, Deterministic)
    # Expected: All algorithms should perform similarly well
    
    print("\n[1/3] GridWorld (5x5)")
    results_gridworld = run_experiment(
        env_class=GridWorld,
        env_kwargs={'size': 5},
        agent_classes=agent_classes,
        agent_kwargs=agent_kwargs,
        n_episodes=n_episodes,
        n_runs=n_runs
    )
    all_results['GridWorld'] = results_gridworld
    plot_learning_curves(results_gridworld, "GridWorld (5x5)", "results_gridworld.png")
    
    
    # EXPERIMENT 2: CliffWalking (Has Dangerous Cliff)
    # Expected: SARSA should be more conservative (avoid cliff)
    #           Q-learning may walk close to cliff (higher risk)
    
    print("\n[2/3] CliffWalking")
    results_cliff = run_experiment(
        env_class=None,
        env_kwargs={'gym_env': 'CliffWalking-v1'},
        agent_classes=agent_classes,
        agent_kwargs=agent_kwargs,
        n_episodes=n_episodes,
        n_runs=n_runs
    )
    all_results['CliffWalking'] = results_cliff
    plot_learning_curves(results_cliff, "CliffWalking", "results_cliffwalking.png")
    
    # EXPERIMENT 3: FrozenLake (Stochastic - Agent Can Slip)
    # Expected: More challenging due to randomness
    # Tests how algorithms handle uncertainty
    
    print("\n[3/3] FrozenLake (Stochastic)")
    results_frozen = run_experiment(
        env_class=None,
        env_kwargs={'gym_env': 'FrozenLake-v1'},
        agent_classes=agent_classes,
        agent_kwargs={'alpha': 0.1, 'gamma': 0.99, 'epsilon': 0.2},  # Higher exploration
        n_episodes=1000,  # More episodes for stochastic environment
        n_runs=n_runs
    )
    all_results['FrozenLake'] = results_frozen
    plot_learning_curves(results_frozen, "FrozenLake (Stochastic)", "results_frozenlake.png")
    
    # SUMMARY: Compare all algorithms across all environments
    print("\n[Summary] Comparing all algorithms across environments")
    plot_comparison_summary(all_results, "results_comparison.png")
    
    # Print final statistics
    print("FINAL RESULTS (Average reward over last 50 episodes)")
    print(f"{'Environment':<15} {'Q-learning':<15} {'SARSA':<15} {'Double Q':<15}")
    
    for env_name, results in all_results.items():
        row = f"{env_name:<15}"
        for agent_name in ['QLearningAgent', 'SARSAAgent', 'DoubleQLearningAgent']:
            if agent_name in results:
                final = np.mean(results[agent_name]['rewards'][:, -50:])
                row += f"{final:<15.2f}"
        print(row)
    
    print("\nDone! Results saved to PNG files.")
