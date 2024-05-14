# Connect-4-Reinforcement-learning

## Background and Motivation

Connect Four is a two players game which takes place on a 6x7 rectangular board placed vertically between them. One player has 21 blue coins and the other 21 red coins. Each player can drop a coin at the top of the board in one of the seven columns; the coin falls down and fills the lower unoccupied square. Of course a player cannot drop a coin in a certain column if it's already full (i.e. if it already contains six coins).

## Problem Statement

The rules of the game are as follows:

- Every player tries to connect their coin type in a sequence of 4.
- After every move by any player, the status of the game is checked whether it is over. A game is considered over in the following scenarios:
  - Either of the players have 4 coins on the board in a sequence vertically, horizontally or diagonally.
  - The board is full that is after 42 moves none of the players got a sequence of 4. In this case the game is a tie.
- Using the above rules, the problem statement is to train an agent who plays the game optimally with a fair understanding of the game's rules and should be able to play with different players ( Human, RandomAgent, SmartAgent, MinmaxAgent ) and maximize its number of wins.

## Methodology

### Environment

We began by implementing the Connect Four environment, which includes the game board, rules, and player interactions as defined above. Next, we defined the state representations, action spaces, and reward functions tailored to the Connect Four game.

### Opponent Agents
We created different agents that will help the RL agent to learn from and play against. 
* Random Agent: An agent that plays any random move given the valid moves it can take.
* Smart Agent: An agent that is better than a Random Agent. It plays randomly, but when it encounters a position where it can win, it chooses that action promptly.
* Minimax Agent: An agent that foresees K next moves and puts the coin in the column that maximises the reward it can get. The reward is allocated based on some configurations. We also optimised this algorithm using alpha-beta pruning.


### Policies
We employed two Reinforcement Learning algorithms: Q-learning and Monte Carlo Tree Search (MCTS).

* Q-Learning - A model-free RL algorithm that updates action-value estimates iteratively using observed rewards, enabling the RL agent to learn optimal policies and improve decision-making over time. The reward will be 1 for winning, -1 for losing, 0.5 for a tie and 0 otherwise. The targets were calculated according to the Q learning algorithm:
Q(s,a) = Q(s,a) + α(max(Q(s’,a’))+gR-Q(s,a)) where Q is the Q function, s is the state, a is the action, α is the learning rate, R is the reward and g is the discount factor.

* MCTS - A heuristic search algorithm that incrementally builds a search tree by simulating random plays from the current game state. It balances exploration and exploitation to guide decision-making, maximizing the agent's chances of winning. ![image](https://github.com/amitmakkad/Connect-4-Reinforcement-Learning/assets/79632719/bbb0390c-931b-4e5f-a1f2-eabfa61784fe)

## Performance

To evaluate the effectiveness of our trained RL agent, we measure its win rate against different Agents. Please refer report for detailed analysis.

## Contributer

* Amit Kumar Makkad - mcts
* Mukul Jain - Q learning
* Nilay Ganvit - Display 

This project is part of course CS432 Reinforcement Learning IIT Indore under guidance of Dr. Surya Prakash.

