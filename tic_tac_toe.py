"""Exploration of RL methods to solve the devilish game known as Tic-Tac-Toe."""

import numpy as np
import sys
import random


class PlayerInterface:
    """Describes how to interact with the game."""

    def move(self, grid):
        """Returns the index to check."""
        abstract


class CLIPlayer:
    """Command-line player (for tests and giggles)."""

    def __init__(self, size=3):
        self.size = size

    def move(self, grid):
        for i in range(self.size):
            print(grid[i*self.size:(i+1)*self.size])
        while True:
            m = input('Choose a move (1-9)> ')
            try:
                if m == 'stop':
                    sys.exit(0)
                return int(m) - 1
            except ValueError:
                print('Invalid move!')
        return int(m)  # not robust lol


def list_actions(grid):
    """Simple function to return all actions in a tictactoe grid."""
    return [i for i, entry in enumerate(grid) if entry==0]


class RandomPlayer:
    """A player picking actions at random (default)."""

    def move(self, grid):
        return random.choice(list_actions(grid))




def tictactoe(player, opponent=RandomPlayer(), size=3):
    """Play a game of tic-tac-toe with a player (with interface).

    This chooses a starting player at random, and iteratively calls
     player.move, then opponent.move, until a player aligns 3 identical
     symbols on the grid (and wins), or the grid is full (draw).

    Returns +1 if `player` wins, -1 if `opponent` wins, 0 if a draw occurs.

    The input `size` is the size of one side of the grid. Currently the
     only supported value is size=3 because I am lazy.
    """
    n = size * size
    # We represent the grid as a list of length n = size^2.
    grid = [0 for i in range(n)]
    # Choose whether the player starts.
    player_to_move = random.randrange(2)
    for i in range(n):  # At most n moves to be made.
        if player_to_move:
            move = player.move(grid)
            entry = 1
        else:
            # Opponent makes a move.
            move = opponent.move(grid)
            entry = -1
        # Make the move: update the grid.
        if grid[move]:
            raise Exception('Illegal move!')
        grid[move] = entry
        # Check if the player wins. (lol dirty only works with size=3)
        # TOFIX: extend to >3.
        if (grid[0] == grid[1] == grid[2] != 0) or \
           (grid[3] == grid[4] == grid[5] != 0) or \
           (grid[6] == grid[7] == grid[8] != 0) or \
           (grid[0] == grid[3] == grid[6] != 0) or \
           (grid[1] == grid[4] == grid[7] != 0) or \
           (grid[2] == grid[5] == grid[8] != 0) or \
           (grid[0] == grid[4] == grid[8] != 0) or \
           (grid[2] == grid[4] == grid[6] != 0):
            # Active player wins!
            return player_to_move * 2 - 1
        # Swap the leading player.
        player_to_move = 1 - player_to_move
    # No winner -- SAD.
    return 0


# A first application -- classical Q-learning.

class classicalQLearning(PlayerInterface):
    """Where we go one, we go all.

    This learns the Q(state, action) function by repeatedly trying random
     actions (in learning mode) to explore the space. Q is then estimated
     as the cumulative reward for the state.

    The estimation of Q is done by choosing actions that are either optimal
     according to the current Q ("on-policy"), or purely random (with probability
     epsilon, 0.5 by default). This ensures convergence to the real value of Q,
     I believe? (I'm not sure how to interpret equation 4.2 and am
     too tired to try).

    Note: this will not scale, obviously, since the number of states explodes
     with the size. That's why we'll need Fitted Q-learning."""

    def __init__(self, size=3, gamma=1, epsilon=0.5):
        self.size = size
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}  # Maps (state, action) to (cumulative_reward, num_samples).
        # Special flag that describes whether to memorise move or not.
        self._training_mode = False


    def move(self, grid):
        """Choose a move to make on the grid."""
        # If in training mode, choose a completely random sample w.p. epsilon
        if self._training_mode and random.random() < self.epsilon:
            # Explore: return random action (and store it).
            choice = self._random_move(grid)
        else:
            choice = self._best_move(grid)
        if self._training_mode:
            self._training_samples.append( (tuple(grid), choice))
        return choice


    def _random_move(self, grid):
        """Do a random move on this grid."""
        return random.choice(list_actions(grid))

    def _best_move(self, grid):
        """Do the best move possible, according to current Q."""
        # Naive implementation of a max over the structure.
        max_so_far = -np.inf
        best_choice = None
        for a in list_actions(grid):
            # In case the state is unknown: return 0.
            cumreward, count = self.Q.get((tuple(grid), a), (0, 1))
            if cumreward / count > max_so_far:
                max_so_far = cumreward / count
                best_choice = a
        return best_choice


    def train(self, epochs=100):
        """Play the game several times, and update the internal Q."""
        self._training_mode = True
        for epoch in range(epochs):
            # Do one full run of the algorithm, and collect
            #  intermediate actions (to assign rewards).
            self._training_samples = []
            reward = tictactoe(self, size=self.size)
            # Now, increase the reward of each Q.
            # Since the only reward is in the last step, we multiply the reward
            #  by scaling gamma^{-step} (although that's probably unnecessary) 
            depth = len(self._training_samples)
            for i, entry in enumerate(self._training_samples):
                cumreward, count = self.Q.get(entry, (0, 0))
                self.Q[entry] = (cumreward+reward*self.gamma**(depth-1-i), count+1)
        # At the end, reset to sicko mode.
        self._training_mode = False


    def evaluate(self, num_repeats=100):
        """Repeatedly play against a random opponent."""
        self._training_mode = False
        return np.mean([tictactoe(self, size=self.size) for _ in range(num_repeats)])




# Fitted Q-learning.

import torch
import torch.optim as optim


class fittedQLearning(classicalQLearning):

    def __init__(self, size=3, gamma=0.5, epsilon=0.5, hidden_layers=[1000]):
        # Initialise the relevant internal variables.
        classicalQLearning.__init__(self, size, gamma, epsilon)
        # Create the fitted Q-learning network (a MLP).
        #  The input is the full grid (+1 = this player, -1 = opponent, 0 = empty).
        #  The output is the Q(s, e) for this move and each state state.
        self.output_size = self.input_size = size * size
        layers = []
        for in_, out_ in zip([self.input_size] + hidden_layers[:-1], hidden_layers):
            layers.append(torch.nn.Linear(in_, out_))
            layers.append(torch.nn.ReLU())
        # Add a final linear layer.
        layers.append(torch.nn.Linear(hidden_layers[-1], self.output_size))
        self.Q = torch.nn.Sequential(*layers)


    # .move is inherited, and only _best_move needs to be implemented.
    def _best_move(self, grid):
        input_ = torch.Tensor([grid])
        with torch.no_grad():
            output_ = self.Q.forward(input_)
        output_ = output_.numpy()[0]
        # Restrict Q(s,a) to *valid* actions.
        valid_actions = list_actions(grid)
        output_ = output_[valid_actions]
        best_idx = np.argmax(output_)
        return valid_actions[best_idx]


    def train(self, epochs=1000, gradient_steps=10):
        self._training_mode = True
        # Run the experiment `epochs` times.
        # For each step in the experiment, save (s, a, r, s'),
        #  (initial state s, action a, reward r, final state s').
        observations = []
        for epoch in range(epochs):
            # Do one full run of the algorithm, and collect
            #  intermediate actions (to assign rewards).
            self._training_samples = []
            final_reward = tictactoe(self, size=self.size)  # Reward in last step only.
            for (s, a), (s_next, _) in zip(self._training_samples[:-1],
                                           self._training_samples[1:]):
                reward = tictactoe(self, size=self.size)
                observations.append( (s, a, 0, s_next) )
            # Last step: absorbant state.
            final_state, final_action = self._training_samples[-1]
            observations.append( (final_state, final_action, reward, final_state) )
        # Unzip the observations.
        batch_states, batch_actions, batch_reward, batch_next_states = zip(*observations)
        batch_size = len(observations)
        batch_states = torch.Tensor(batch_states)
        batch_reward = torch.Tensor(batch_reward)
        # Compute the target (Y). This implies making a forward pass on all
        #  *next* states encountered (to estimate Q(s', a', thetak)).
        # First, compute the reward (no grad!).
        with torch.no_grad():
            Q_on_actions = self.Q.forward(torch.Tensor(batch_next_states))  # [Q(s', a') for a' in A].
            # Y = r + gamma * max_{a'} Q(s', a', thetak)  (4.3, pp. 26).
            # But we must restrict to the set of legal actions (so we iterate).
            Q_on_actions = Q_on_actions.numpy()
            batch_target = np.zeros((batch_size,))
            for i, s in enumerate(batch_next_states):
                batch_target[i] = Q_on_actions[i,list_actions(s)].max()
            batch_target = batch_reward + self.gamma * torch.Tensor(batch_target)
        # Note that we perform the target once, and don't update its value during
        #  the gradient descent steps. This is an optimization that improves the method.
        # Do a few steps of gradient descent.
        optimizer = optim.SGD(self.Q.parameters(), lr=0.01)
        for i in range(gradient_steps):
            # Reset the gradients stored in parameters.
            optimizer.zero_grad()
            # Compute Q(s, a) for each state encountered and *all* actions.
            batch_Q = self.Q.forward(batch_states)
            # Then, for each state, select the action actually performed b
            batch_Q_sa = batch_Q[np.arange(batch_size), batch_actions]  # Extract Q(s,a).
            loss = torch.mean( (batch_Q_sa - batch_target) ** 2 )
            print(f'\r\tIter {i+1}: loss = {loss.item()}', end='')
            loss.backward()
            optimizer.step()
        print()
        # At the end, reset to sicko mode.
        self._training_mode = False

    # .evaluate is inherited from super().




if __name__ == '__main__':
    ## Uncomment this to try classical Q-learning:
    # learner = classicalQLearning(gamma=0.5)
    # max_iter = 400
    ## Uncomment this to try fitted Q-learning:
    learner = fittedQLearning(gamma=0.5, epsilon=0.5, hidden_layers=[10000])
    max_iter = 50
    score_l = []  # List of scores, for plotting purposes.
    for _ in range(max_iter):
        learner.train()
        score = learner.evaluate(1000)
        print(f'[{_}] Score = {score}')
        score_l.append(score)
    # Nice plot, just for you.
    import matplotlib.pyplot as plt
    plt.plot(score_l)
    plt.xlabel('Number of epochs')
    plt.ylabel('Expected reward')
    plt.show()
    # Play against the AI!
    while True:
        print('\nPlay against the AI!')
        result = tictactoe(learner, CLIPlayer())
        print({
            +1: 'The AI wins!',
            -1: 'You win!',
            0: "It's a draw."}[result])