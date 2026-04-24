"""
Q-Learning Implementation for GridWorld
CS5100 FAI Capstone - Milestone 1

This is an implementation of the Q-learning algorithm as described in
Watkins & Dayan (1992) "Q-Learning" paper.

Q-learning is a model-free reinforcement learning algorithm that learns
the value of actions in states without needing a model of the environment.

Key concepts:
- Q-value: Q(s,a) = expected total reward starting from state s, taking action a
- Q-table: A table storing Q-values for all state-action pairs
- Update rule: Q(s,a) = Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]

"""

import numpy as np          # NumPy: library for numerical calculations (arrays, matrices)
import matplotlib.pyplot as plt  # Matplotlib: library for creating plots and visualizations
from typing import Tuple, List   
import random                    # For random number generation (exploration)


# ENVIRONMENT CLASS

class GridWorld:
    """
    GridWorld Environment - A simple grid-based environment for testing RL algorithms.
    
    The environment is a 5x5 grid:
    - Agent starts at top-left corner (0,0)
    - Goal is at bottom-right corner (4,4)
    - Agent can move: up, down, left, right
    - Agent receives reward of +1 for reaching goal, -0.01 for each step
    
    Visual representation:
        (0,0) (0,1) (0,2) (0,3) (0,4)     S = Start
        (1,0) (1,1) (1,2) (1,3) (1,4)     G = Goal
        (2,0) (2,1) (2,2) (2,3) (2,4)     
        (3,0) (3,1) (3,2) (3,3) (3,4)     Optimal path: 8 steps
        (4,0) (4,1) (4,2) (4,3) (4,4)     (4 right + 4 down, or 4 down + 4 right)
    
    Why 25 states? 5 rows × 5 columns = 25 possible positions
    Why 4 actions? up(0), down(1), left(2), right(3)
    """
    
    def __init__(self, size: int = 5):
        """
        Initialize the GridWorld environment.
        
        Args:
            size: The size of the grid (default 5x5)
        """
        self.size = size                      # Grid is size × size
        self.state = (0, 0)                   # Current position, starts at top-left
        self.goal = (size - 1, size - 1)     # Goal position at bottom-right
        
        # Define the 4 possible actions
        # 0 = up, 1 = down, 2 = left, 3 = right
        self.actions = [0, 1, 2, 3]
        self.action_names = ['up', 'down', 'left', 'right']
        
    def reset(self) -> Tuple[int, int]:
        """
        Reset the environment to starting state.
        Called at the beginning of each episode.
        
        Returns:
            Starting state (0, 0)
        """
        self.state = (0, 0)  # Put agent back at start
        return self.state
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute one action in the environment.
        
        This is the main function that simulates the environment dynamics.
        
        Args:
            action: Which direction to move (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            next_state: The new position after taking the action
            reward: The reward received (+1 for goal, -0.01 otherwise)
            done: Whether the episode is finished (reached goal)
        """
        row, col = self.state  # Current position
        
        # Apply the action (with boundary checking)
        if action == 0:    # UP: decrease row (but can't go below 0)
            row = max(0, row - 1)
        elif action == 1:  # DOWN: increase row (but can't exceed size-1)
            row = min(self.size - 1, row + 1)
        elif action == 2:  # LEFT: decrease column (but can't go below 0)
            col = max(0, col - 1)
        elif action == 3:  # RIGHT: increase column (but can't exceed size-1)
            col = min(self.size - 1, col + 1)
        
        # Update the current state
        self.state = (row, col)
        
        # Check if we reached the goal
        done = (self.state == self.goal)
        
        # Calculate reward:
        # +1.0 for reaching goal (encourages reaching the goal)
        # -0.01 for each step (encourages finding shortest path)
        reward = 1.0 if done else -0.01
        
        return self.state, reward, done
    
    def get_num_states(self) -> int:
        """
        Get total number of states in the environment.
        For a 5x5 grid: 5 × 5 = 25 states
        """
        return self.size * self.size
    
    def get_num_actions(self) -> int:
        """
        Get total number of possible actions.
        4 actions: up, down, left, right
        """
        return len(self.actions)
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """
        Convert 2D position (row, col) to 1D index for Q-table.
        
        Example for 5x5 grid:
            (0,0) -> 0,  (0,1) -> 1,  (0,2) -> 2,  (0,3) -> 3,  (0,4) -> 4
            (1,0) -> 5,  (1,1) -> 6,  (1,2) -> 7,  (1,3) -> 8,  (1,4) -> 9
            ... and so on
        
        Formula: index = row × grid_size + column
        """
        return state[0] * self.size + state[1]
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """
        Convert 1D index back to 2D position (row, col).
        
        Formula: row = index // grid_size, col = index % grid_size
        """
        return (index // self.size, index % self.size)


# Q-LEARNING AGENT CLASS

class QLearningAgent:
    """
    Q-Learning Agent - Implements the Q-learning algorithm from Watkins & Dayan (1992).
    
    Key components:
    1. Q-table: Stores Q(s,a) values for all state-action pairs
    2. Epsilon-greedy policy: Balances exploration vs exploitation
    3. Update rule: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
    
    Hyperparameters:
    - alpha (learning rate): How much to update Q-values (0.1 = update 10% towards target)
    - gamma (discount factor): How much to value future rewards (0.99 = value future highly)
    - epsilon (exploration rate): Probability of random action (0.1 = 10% random)
    """
    
    def __init__(
        self, 
        num_states: int,          # Total number of states (25 for 5x5 grid)
        num_actions: int,         # Total number of actions (4: up/down/left/right)
        learning_rate: float = 0.1,    # alpha: how fast to learn
        discount_factor: float = 0.99, # gamma: importance of future rewards
        epsilon: float = 0.1           # exploration rate
    ):
        """
        Initialize the Q-learning agent.
        
        Args:
            num_states: Number of states in the environment
            num_actions: Number of possible actions
            learning_rate: Alpha - controls how much Q-values change per update
            discount_factor: Gamma - controls importance of future vs immediate rewards
            epsilon: Probability of choosing random action for exploration
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate      # Learning rate (typically 0.01 to 0.5)
        self.gamma = discount_factor    # Discount factor (typically 0.9 to 0.99)
        self.epsilon = epsilon          # Exploration rate (typically 0.1 to 0.3)
        
        # Initialize Q-table with zeros
        # Q-table shape: (num_states × num_actions) = (25 × 4) for our GridWorld
        # Initially all Q-values are 0, meaning agent has no knowledge
        self.q_table = np.zeros((num_states, num_actions))
        
    def choose_action(self, state_index: int) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Epsilon-greedy balances:
        - Exploration: Try random actions to discover new things (epsilon probability)
        - Exploitation: Use what we learned to get best reward (1-epsilon probability)
        
        Args:
            state_index: Current state (as 1D index)
            
        Returns:
            action: The chosen action (0-3)
        """
        # Generate random number between 0 and 1
        if random.random() < self.epsilon:
            # EXPLORATION: With probability epsilon, choose random action
            # This helps discover potentially better actions we haven't tried
            return random.randint(0, self.num_actions - 1)
        else:
            # EXPLOITATION: With probability (1-epsilon), choose best known action
            # np.argmax finds the action with highest Q-value for this state
            return np.argmax(self.q_table[state_index])
    
    def update(
        self, 
        state: int,       # Current state index
        action: int,      # Action taken
        reward: float,    # Reward received
        next_state: int,  # Next state index
        done: bool        # Whether episode ended
    ):
        """
        Update Q-value using the Q-learning update rule.
        
        This is THE CORE of Q-learning algorithm from the paper!
        
        Update formula:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        
        Breaking it down:
        - Q(s,a): Current Q-value estimate
        - r: Immediate reward received
        - gamma * max Q(s',a'): Discounted future value (best possible from next state)
        - alpha: Learning rate (how much to adjust)
        
        The term [r + gamma * max Q(s',a')] is the "target" - what Q should be
        The term [target - Q(s,a)] is the "TD error" - how wrong our estimate was
        
        Args:
            state: Current state index
            action: Action that was taken
            reward: Reward received after taking action
            next_state: State we ended up in
            done: Whether the episode ended
        """
        # Get current Q-value estimate for this state-action pair
        current_q = self.q_table[state, action]
        
        if done:
            # If episode ended (reached goal), there's no future reward
            # Target is just the immediate reward
            target = reward
        else:
            # If episode continues, target includes future value
            # We use the MAX Q-value of next state (this is what makes it "Q-learning")
            # This is different from SARSA which uses the actual next action's Q-value
            max_next_q = np.max(self.q_table[next_state])  # Best possible Q from next state
            target = reward + self.gamma * max_next_q      # r + gamma * max Q(s',a')
        
        # Calculate TD error (Temporal Difference error)
        # This measures how "surprised" we are by the actual outcome
        td_error = target - current_q
        
        # Update the Q-value
        # New Q = Old Q + learning_rate × TD_error
        # If td_error > 0: Q-value was too low, increase it
        # If td_error < 0: Q-value was too high, decrease it
        self.q_table[state, action] = current_q + self.alpha * td_error
    
    def get_policy(self) -> np.ndarray:
        """
        Extract the greedy policy from Q-table.
        
        For each state, find the action with the highest Q-value.
        This represents what the agent thinks is the best action in each state.
        
        Returns:
            Array of best actions for each state
        """
        # np.argmax with axis=1 returns the column (action) with max value for each row (state)
        return np.argmax(self.q_table, axis=1)


# TRAINING FUNCTION

def train_qlearning(
    env: GridWorld, 
    agent: QLearningAgent, 
    num_episodes: int = 500,    # Train for 500 episodes
    max_steps: int = 200,       # Safety cap to avoid rare non-terminating episodes
) -> List[float]:
    """
    Train the Q-learning agent by running multiple episodes.
    
    One episode = agent starts at (0,0), takes actions until reaching goal.
    
    Training loop:
    1. Reset environment to start state
    2. Repeat until episode ends:
       a. Choose action (epsilon-greedy)
       b. Take action, observe reward and next state
       c. Update Q-value
       d. Move to next state
    3. Record total reward for this episode
    
    Args:
        env: The GridWorld environment
        agent: The Q-learning agent
        num_episodes: How many episodes to train
        
    Returns:
        List of total rewards for each episode (to plot learning curve)
    """
    episode_rewards = []  # Track rewards to see learning progress
    
    # Run training for specified number of episodes
    for episode in range(num_episodes):
        # Start a new episode
        state = env.reset()                    # Reset to starting position (0,0)
        state_idx = env.state_to_index(state)  # Convert to 1D index for Q-table
        total_reward = 0                       # Track total reward this episode
        done = False                           # Episode not finished yet
        
        # Run until episode ends (agent reaches goal) or hits safety cap
        steps = 0
        while not done and steps < max_steps:
            # Step 1: Choose action using epsilon-greedy policy
            action = agent.choose_action(state_idx)
            
            # Step 2: Take action in environment
            next_state, reward, done = env.step(action)
            next_state_idx = env.state_to_index(next_state)
            
            # Step 3: Update Q-value based on experience
            # This is where learning happens!
            agent.update(state_idx, action, reward, next_state_idx, done)
            
            # Step 4: Move to next state
            state_idx = next_state_idx
            total_reward += reward
            steps += 1
        
        # Episode finished - record the total reward
        episode_rewards.append(total_reward)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])  # Average of last 100
            print(f"Episode {episode + 1}: Average reward (last 100) = {avg_reward:.2f}")
    
    return episode_rewards


# VISUALIZATION FUNCTION

def visualize_results(
    env: GridWorld, 
    agent: QLearningAgent, 
    episode_rewards: List[float]
):
    """
    Create visualizations to understand what the agent learned.
    
    Creates 3 plots:
    1. Learning Curve: Shows how reward improves over training
    2. Q-value Heatmap: Shows value of each state (brighter = higher value)
    3. Learned Policy: Shows what action to take in each state (arrows)
    """
    # Create a figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # PLOT 1: Learning Curve
    # Shows total reward per episode over training
    # Good learning = curve goes up and stabilizes
    axes[0].plot(episode_rewards)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Learning Curve')
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)  # Reference line at y=0
    
    # PLOT 2: Q-value Heatmap
    # Shows max Q-value for each state
    # Brighter color = higher value = closer to goal
    # Get max Q-value for each state and reshape to grid
    max_q = np.max(agent.q_table, axis=1).reshape(env.size, env.size)
    im = axes[1].imshow(max_q, cmap='viridis')  # viridis: purple=low, yellow=high
    axes[1].set_title('Max Q-value per State')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im, ax=axes[1])  # Add color legend
    
    # Mark start (green circle) and goal (red star)
    axes[1].plot(0, 0, 'go', markersize=10, label='Start')
    axes[1].plot(env.size-1, env.size-1, 'r*', markersize=15, label='Goal')
    axes[1].legend()
    
    # PLOT 3: Learned Policy (arrows showing best action in each state)
    # This is what the agent learned: "in state X, do action Y"
    policy = agent.get_policy().reshape(env.size, env.size)
    arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}  # Map action numbers to arrows
    
    axes[2].imshow(np.zeros((env.size, env.size)), cmap='gray', alpha=0.1)
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) == env.goal:
                # Mark the goal with 'G'
                axes[2].text(j, i, 'G', ha='center', va='center', fontsize=14, color='red')
            else:
                # Show arrow for best action in this state
                axes[2].text(j, i, arrows[policy[i, j]], ha='center', va='center', fontsize=14)
    
    axes[2].set_title('Learned Policy')
    axes[2].set_xlabel('Column')
    axes[2].set_ylabel('Row')
    axes[2].set_xticks(range(env.size))
    axes[2].set_yticks(range(env.size))
    axes[2].grid(True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('qlearning_results.png', dpi=150)
    plt.close()
    print("Results saved to qlearning_results.png")



# TESTING FUNCTION


def test_learned_policy(env: GridWorld, agent: QLearningAgent, num_tests: int = 10):
    """
    Test the learned policy WITHOUT exploration.
    
    After training, we want to see how well the agent performs
    when it always picks the best action (no random exploration).
    
    Args:
        env: The GridWorld environment
        agent: The trained Q-learning agent
        num_tests: How many test runs to do
    """
    print("\nTesting learned policy...")
    
    # Turn off exploration for testing
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Always pick best action, no randomness
    
    total_rewards = []
    total_steps = []
    
    # Run multiple tests
    for _ in range(num_tests):
        state = env.reset()
        state_idx = env.state_to_index(state)
        done = False
        episode_reward = 0
        steps = 0
        
        # Run until goal reached (with max steps to prevent infinite loop)
        while not done and steps < 100:
            action = agent.choose_action(state_idx)  # Now always picks best action
            state, reward, done = env.step(action)
            state_idx = env.state_to_index(state)
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    # Print results
    print(f"Average reward: {np.mean(total_rewards):.2f} (+/- {np.std(total_rewards):.2f})")
    print(f"Average steps to goal: {np.mean(total_steps):.1f} (+/- {np.std(total_steps):.1f})")
    
    # Compare to optimal
    # Optimal path for 5x5 grid: 4 right + 4 down = 8 steps
    print(f"Optimal path length for {env.size}x{env.size} grid: {2*(env.size-1)} steps")


# MAIN PROGRAM

if __name__ == "__main__":
    """
    Main program - runs when you execute: python qlearning_gridworld.py
    
    Steps:
    1. Create environment and agent
    2. Train the agent
    3. Test the learned policy
    4. Visualize results
    5. Print Q-table sample
    """
    
    print("Q-Learning Reproduction - CS5100 Capstone")
    print("Based on Watkins & Dayan (1992)")
    
    # STEP 1: Create environment and agent
    grid_size = 5  # 5x5 grid = 25 states
    
    # Create the GridWorld environment
    env = GridWorld(size=grid_size)
    
    # Create the Q-learning agent with hyperparameters
    agent = QLearningAgent(
        num_states=env.get_num_states(),     # 25 states
        num_actions=env.get_num_actions(),   # 4 actions
        learning_rate=0.1,    # alpha: update Q-values by 10% towards target
        discount_factor=0.99, # gamma: highly value future rewards
        epsilon=0.1           # 10% random exploration, 90% exploitation
    )
    
    # Print configuration
    print(f"\nEnvironment: {grid_size}x{grid_size} GridWorld")
    print(f"Number of states: {env.get_num_states()}")
    print(f"Number of actions: {env.get_num_actions()}")
    print(f"\nHyperparameters:")
    print(f"  Learning rate (alpha): {agent.alpha}")
    print(f"  Discount factor (gamma): {agent.gamma}")
    print(f"  Exploration rate (epsilon): {agent.epsilon}")
    
    
    # STEP 2: Train the agent
    print("\nTraining...")
    episode_rewards = train_qlearning(env, agent, num_episodes=500)
    
    # STEP 3: Test the learned policy
    test_learned_policy(env, agent)
    
    
    # STEP 4: Visualize results
    print("\nGenerating visualizations...")
    visualize_results(env, agent, episode_rewards)
    
    # STEP 5: Print Q-table sample
    print("\nQ-table sample (first 5 states):")
    print("State | Up    | Down  | Left  | Right")
    print("-" * 45)
    for i in range(min(5, env.get_num_states())):
        state = env.index_to_state(i)
        q_vals = agent.q_table[i]
        print(f"{state}  | {q_vals[0]:5.2f} | {q_vals[1]:5.2f} | {q_vals[2]:5.2f} | {q_vals[3]:5.2f}")
    
    print("\nTraining complete! Check qlearning_results.png for visualizations.")
