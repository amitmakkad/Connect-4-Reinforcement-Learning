from collections import defaultdict
import numpy as np
import random
import pickle
from matplotlib import pyplot as plt
import math

# Class code for MCTS Node................................................................................
class Node:
    def __init__(self, state, winning, move, parent):
        self.parent = parent
        self.move = move
        self.win = 0
        self.games = 0
        self.children = None
        self.state = state
        self.winner = winning

    def set_children(self, children):
        self.children = children

    def get_uct(self):
        if self.games == 0:
            return None
        return (self.win/self.games) + np.sqrt(2*np.log(self.parent.games)/self.games)


    def select_move(self):
        """
        Select best move and advance
        :return:
        """
        if self.children is None:
            return None, None

        winners = [child for child in self.children if child.winner]
        if len(winners) > 0:
            return winners[0], winners[0].move

        games = [child.win/child.games if child.games > 0 else 0 for child in self.children]
        best_child = self.children[np.argmax(games)]
        return best_child, best_child.move


    def get_children_with_move(self, move):
        if self.children is None:
            return None
        for child in self.children:
            if child.move == move:
                return child

        raise Exception('Not existing child')


# some play functions.....................................................................................
def create_grid(sizeX=6, sizeY=7):
    return np.zeros((sizeX, sizeY), dtype=int)

def reset(grid):
    return np.zeros(grid.shape, dtype=int)

def play(grid_, column, player=None):
    """
    Play at given column, if no player provided, calculate which player must play, otherwise force player to play
    Return new grid and winner
    """
    grid = grid_.copy()
    if player is None:
        player = get_player_to_play(grid)

    if can_play(grid, column):
        row = grid.shape[0] - 1 - np.sum(np.abs(grid[:, column]), dtype=int)
        grid[row, column] = player
    else:
        raise Exception('Error : Column {} is full'.format(column))
    return grid, player if has_won(grid, player, row, column) else 0


def can_play(grid, column):
    """
    Check if the given column is free
    """
    return np.sum(np.abs(grid[:, column])) < len(grid[:, column])

def valid_move(grid):
    return [i for i in range(grid.shape[1]) if can_play(grid, i)]

def has_won(grid, player, row, column):
    """
    Check if player has won with is new piece
    """
    player += 1
    grid += 1
    row_str = ''.join(grid[row, :].astype(str).tolist())
    col_str = ''.join(grid[:, column].astype(str).tolist())
    up_diag_str = ''.join(np.diagonal(grid, offset=(column - row)).astype(str).tolist())
    down_diag_str = ''.join(np.diagonal(np.rot90(grid), offset=-grid.shape[1] + (column + row) + 1).astype(str).tolist())

    grid -= 1
    victory_pattern = str(player)*4
    if victory_pattern in row_str:
        return True
    if victory_pattern in col_str:
        return True
    if victory_pattern in up_diag_str:
        return True
    if victory_pattern in down_diag_str:
        return True

    return False

def get_player_to_play(grid):
    """
    Get player to play given a grid
    """
    player_1 = 0.5 * np.abs(np.sum(grid-1))
    player_2 = 0.5 * np.sum(grid + 1)

    if player_1 > player_2:
        return 1
    else:
        return -1


def to_state(grid):
    grid += 1
    res = ''.join(grid.astype(str).flatten().tolist())
    grid -=1
    return res

def utils_print(grid):
    print_grid = grid.astype(str)
    print_grid[print_grid == '-1'] = 'X'
    print_grid[print_grid == '1'] = 'O'
    print_grid[print_grid == '0'] = ' '
    res = str(print_grid).replace("'", "")
    res = res.replace('[[', '[')
    res = res.replace(']]', ']')
    print(' ' + res)
    print('  ' + ' '.join('0123456'))

def play_(grid, column, player):
    for r in range(5, -1, -1):
        if grid[r][column] == 0:
            grid[r][column] = player
            break

def get_valid_moves(grid):
    return [col for col in range(7) if grid[0][col] == 0]

def is_draw(grid):
    return all(grid[0][c] != 0 for c in range(7))

def check_winner(grid, player):
    for r in range(6):
        for c in range(7):
            if check_direction(grid, r, c, player):
                return True
    return False

def check_direction(grid, row, col, player):
    # Check horizontally
    if col + 3 < 7 and all(grid[row][col + i] == player for i in range(4)):
        return True
    # Check vertically
    if row + 3 < 6 and all(grid[row + i][col] == player for i in range(4)):
        return True
    # Check diagonally (positive slope)
    if row + 3 < 6 and col + 3 < 7 and all(grid[row + i][col + i] == player for i in range(4)):
        return True
    # Check diagonally (negative slope)
    if row - 3 >= 0 and col + 3 < 7 and all(grid[row - i][col + i] == player for i in range(4)):
        return True
    return False

# Different Agents....................................................................................

def random_agent_move(valid_moves):
    return random.choice(valid_moves)


def smart_agent_move(grid, valid_moves):
    for move in valid_moves:
        new_grid, winner = play(grid.copy(), move)
        if winner == 1:
            return move
    return random.choice(valid_moves)


def minimax_agent_move(grid, player):
    action, val = minimax(grid, 2, -math.inf, math.inf, player == 1)
    return action


def minimax(grid, depth, alpha, beta, maximizing_player):
    valid_moves = get_valid_moves(grid)
    if check_winner(grid, 1):
        return (None, 100000)
    elif check_winner(grid, -1):
        return (None, -100000)
    elif is_draw(grid):
        return (None, 0)
    elif depth == 0:
        return (None, evaluate_score(grid))

    if maximizing_player:
        val = -math.inf
        column = random.choice(valid_moves)
        for col in valid_moves:
            grid_copy = [row[:] for row in grid]
            play_(grid_copy, col, 1)
            new_score = minimax(grid_copy, depth - 1, alpha, beta, False)[1]
            if new_score > val:
                val = new_score
                column = col
            alpha = max(new_score, alpha)
            if beta <= alpha:
                break
        return column, val
    else:
        val = math.inf
        column = random.choice(valid_moves)
        for col in valid_moves:
            grid_copy = [row[:] for row in grid]
            play_(grid_copy, col, -1)
            new_score = minimax(grid_copy, depth - 1, alpha, beta, True)[1]
            if new_score < val:
                val = new_score
                column = col
            beta = min(beta, new_score)
            if beta <= alpha:
                break
        return column, val

def evaluate_score(grid):
    score = 0
    for r in range(6):
        for c in range(4):
            window = grid[r][c:c+4]
            score += evaluate_window(window)
    for c in range(7):
        for r in range(3):
            window = [grid[r+i][c] for i in range(4)]
            score += evaluate_window(window)

    for r in range(3):
        for c in range(4):
            window = [grid[r+i][c+i] for i in range(4)]
            score += evaluate_window(window)

    for r in range(3):
        for c in range(3, 7):
            window = [grid[r+i][c-i] for i in range(4)]
            score += evaluate_window(window)

    return score

def evaluate_window(window):
    score = 0
    player_count = np.count_nonzero(window == 1)
    empty_count = np.count_nonzero(window == 0)
    opponent_count = np.count_nonzero(window == -1)

    if player_count == 4:
        score += 100
    elif player_count == 3 and empty_count == 1:
        score += 5
    elif player_count == 2 and empty_count == 2:
        score += 2

    if opponent_count == 3 and empty_count == 1:
        score -= 4
    return score




# Training Code........................................................................................
def train_mcts_during(mcts, training_time):
    cnt=0
    while cnt<training_time:
        mcts=train_mcts_once(mcts)
        cnt+=1
    return mcts

def train_mcts_once(mcts=None):

    if mcts is None:
        mcts = Node(create_grid(), 0, None,  None)

    node = mcts

    # selection
    while node.children is not None:
        # Select highest uct
        ucts = [child.get_uct() for child in node.children]
        if None in ucts:
            node = random.choice(node.children)
        else:
            node = node.children[np.argmax(ucts)]

    # expansion
    moves = valid_move(node.state)
    if len(moves) > 0:

        if node.winner == 0:

            states = [(play(node.state, move), move) for move in moves]
            node.set_children([Node(state_winning[0], state_winning[1], move=move, parent=node) for state_winning, move in states])
            # simulation
            winner_nodes = [n for n in node.children if n.winner]
            if len(winner_nodes) > 0:
                node = winner_nodes[0]
                victorious = node.winner
            else:
                node = random.choice(node.children)
                victorious = random_play_improved(node.state)
        else:
            victorious = node.winner

        # backpropagation
        parent = node
        while parent is not None:
            parent.games += 1
            if victorious != 0 and get_player_to_play(parent.state) != victorious:
                parent.win += 1
            parent = parent.parent


    else:
        print('no valid moves, expended all')

    return mcts

def random_play_improved(grid):   # part of simulation, in which we play randomly to win

    def get_winning_moves(grid, moves, player):
        return [move for move in moves if play(grid, move, player=player)[1]]

    # If can win, win
    while True:
        moves = valid_move(grid)
        if len(moves) == 0:
            return 0
        player_to_play = get_player_to_play(grid)

        winning_moves = get_winning_moves(grid, moves, player_to_play)
        loosing_moves = get_winning_moves(grid, moves, -player_to_play)

        if len(winning_moves) > 0:
            selected_move = winning_moves[0]
        elif len(loosing_moves) == 1:
            selected_move = loosing_moves[0]
        else:
            selected_move = random.choice(moves)
        grid, winner = play(grid, selected_move)
        if np.abs(winner) > 0:
            return player_to_play

#  Playing MCTS with different Agents......................................................................
def save_mcts(mcts, filename):
    with open(filename, 'wb') as f:
        pickle.dump(mcts, f)


def load_mcts(filename):
    with open(filename, 'rb') as f:
        mcts = pickle.load(f)
    return mcts

if __name__ == '__main__':
                                    # intialzing mcts............................................
    mcts = None
    for i in range(10000):
        mcts = train_mcts_once(mcts)

    save_mcts(mcts, 'Bot/mcts_tree1.pkl')
    print('training finished')

    mcts = load_mcts('Bot/mcts_tree.pkl')

                                    # playing against different agents...............................
    num_rounds=10
    num_games = 100
    win_rate=[]
    for kk in range(num_rounds):
        mcts = load_mcts('Bot/mcts_tree.pkl')
        mcts_wins = 0
        opp_wins = 0
        
        for _ in range(num_games):
            grid = create_grid()
            round = 0
            training_time = 4
            node = mcts
            while True:
                if (round % 2) == 0:
                    # MCTS agent's turn
                    new_node, move = node.select_move()
                    if move is None:
                        break
                    if move is not can_play(grid, move):
                        move = random.choice(valid_move(grid))
                    node = train_mcts_during(node, training_time)
                    
                else:
                    # Opponent agent's turn
                    moves = valid_move(grid)
                    if len(moves) == 0:
                        move=None
                        break
                    move = random_agent_move(moves)
                    # move = smart_agent_move(grid,moves)
                    # move = minimax_agent_move(grid, 1)
                   
                winner=0
                if move is not None:
                    grid, winner = play(grid, move)

                if winner != 0:
                    if winner == 1:
                        opp_wins += 1
                    else:
                        mcts_wins += 1
                    break

                round += 1
            mcts=node
        
        save_mcts(mcts,'Bot/mcts_tree.pkl')
        print("after ",kk+1,"00 rounds:")
        print(f"MCTS wins: {mcts_wins}")
        print(f"MinMax Agent wins: {opp_wins}")
        win_rate.append(mcts_wins/100)
                                                # play with human.....................................
    while True:
        grid = create_grid()
        round = 0
        training_time = 5
        node = mcts
        utils_print(grid)
        while True:
            if (round % 2) == 0:
                move = int(input())
                new_node = node.get_children_with_move(move)
                node = train_mcts_during(node, training_time).get_children_with_move(move)
            else:
                new_node, move = node.select_move()
                node = train_mcts_during(node, training_time)
                node, move = node.select_move()

            grid, winner = play(grid, move)

            utils_print(grid)


            assert np.sum(node.state - grid) == 0, node.state
            if winner != 0:
                print('Winner : ', 'X' if winner == -1 else 'O')
                break
            round += 1
