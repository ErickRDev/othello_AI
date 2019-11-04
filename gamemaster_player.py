import time

from models.move import Move
from copy import deepcopy

""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
"""                       AUXILIARY STRUCTURES                           """
""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

EMPTY_CELL = "."

CORNERS = [
    (1, 1),   # north-western corner
    (1, 8),   # north-eastern corner
    (8, 8),   # south-eastern corner
    (8, 1)    # south-western corner
]

DIRECTIONS = [
    (-1, -1), # north-west
    (-1, 0),  # north
    (-1, 1),  # north-east
    (0, 1),   # east
    (1, 1),   # south-east
    (1, 0),   # south
    (1, -1),  # south-west
    (0, -1)   # west
]

total_branches = 0
visited_nodes = 0

""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
"""                           PLAYING LOGIC                              """
""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

class GameMaster:
    """ 
        Simulates an inteligent player
    """
    def __init__(self, color, allowed_round_time=3000):
        self.color = color
        self.opponent_color = "o" if self.color == "@" else "@"

        self.allowed_round_time = allowed_round_time

        self.turn = 1 if self.color == "@" else 2
        self.first_turn = True


    def evaluate_board_configuration(self, board):
        """
            Evaluates the current board configuration based on the linear combination
                of the heuristics described below. The intuition for the heuristics was 
                taken from this article:
                https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA/Final_Paper.pdf

            The heuristics under employment are:

            * Coin Parity
            * Corners
            * Stability
            * Mobility

            @params:
                board: the board configuration to be evaluated;

            @returns:
                A score.
        """
        # counters for coin parity heuristic
        max_coins, min_coins = 0, 0
        # counters for potential mobility heuristic
        max_potential_moves, min_potential_moves = set(), set()
        # counters for corners heuristic
        max_corners_captured, min_corners_captured = 0, 0
        max_potential_corners, min_potential_corners = 0, 0
        max_unlikely_corners, min_unlikely_corners = 0, 0

        # iterating over the board and evaluating current configuration of coins:
        for i in range(1, 9):
            for j in range(1, 9):
                coin = board[i][j]
                if coin == self.color:
                    # coin parity
                    max_coins += 1

                    # potential mobility
                    for direction in DIRECTIONS:
                        # TODO: calculate stability score
                        _i, _j = i + direction[0], j + direction[1]
                        if board[_i][_j] == "." and (_i, _j) not in max_potential_moves:
                            max_potential_moves.add((_i, _j))

                elif coin == self.opponent_color:
                    # coin parity
                    min_coins += 1

                    # potential mobility
                    for direction in DIRECTIONS:
                        _i, _j = i + direction[0], j + direction[1]
                        if board[_i][_j] == "." and (_i, _j) not in max_potential_moves:
                            max_potential_moves.add((_i, _j))

        max_potential_moves_cnt = len(max_potential_moves)
        min_potential_moves_cnt = len(min_potential_moves)

        # calculating coin parity score
        coin_parity_score = 100 * (max_coins - min_coins) / (max_coins + min_coins)

        mobility_score = 0
        # calculating potential mobility score
        if max_potential_moves_cnt + min_potential_moves_cnt != 0:
            mobility_score = (100 * 
                                       ((max_potential_moves_cnt - min_potential_moves_cnt) / 
                                       (max_potential_moves_cnt + min_potential_moves_cnt)))

        # evaluating corners
        for corner_coords in CORNERS:
            corner = board[corner_coords[0]][corner_coords[1]]
            if corner == self.color:
                max_corners_captured += 1

            if corner == self.opponent_color:
                min_corners_captured += 1

            # TODO: evaluate potential corner captures
            # TODO: evaluate unlikely corner captures

        corners_score = 0
        # calculating corner scores
        if max_corners_captured + min_corners_captured != 0:
            corners_score = (100 * 
                            ((max_corners_captured - min_corners_captured)) / 
                            ((max_corners_captured + min_corners_captured)))

        return coin_parity_score + mobility_score + corners_score


    def minimax(self, board, should_maximize, depth_level, max_depth, alpha, beta):
        """
            Implementation of the minimax algorithm with alpha-beta pruning.

            @intuition:
            In general, the minimax algorithm has a search strategy and heuristics
                that are used to evaluate a given configuration of the board.
                The idea here is to use a depth-first search strategy until
                we reach the desired depthness level, at which point we evaluate
                the board configuration as if we were on a leaf node,
                calculating a "score". This score is then propagated up the tree
                following the strategy of each node - to minimize or to maximize -
                while also applying alpha-beta pruning to optimize max depth reach.

            @params:
                board           : the current board configuration;
                should_maximize : whether this node should maximize or minimize the score;
                depth_level     : current depth-level in the search;
                max_depth       : the max depth that we're allowed to go in the search;
                alpha           : the highest-score we've found so far along current branch;
                beta            : the lowest-score we've found so far along current branch;

            @returns:
                The best action (movement) found by the algorithm;
        """
        if depth_level == max_depth:
            # recursion max depth reached; treating this node as if it was a leaf-node
            score = self.evaluate_board_configuration(board)
            return (None, score)

        if should_maximize:
            # if we're maximizing, this is our turn,
            #   as we always want to maximize our score;
            color = self.color
            target_color = self.opponent_color
        else:
            # if we're minimizing, this is our opponent's turn,
            #   as we always want to minimize his/her score;
            color = self.opponent_color
            target_color = self.color

        # TODO: apply alpha-beta pruning

        found_move = False
        possible_moves = []

        for i in range(1, 9):
            for j in range(1, 9):
                if board[i][j] == EMPTY_CELL:

                    flanked_coins = []
                    found_flank = False

                    for direction in DIRECTIONS:
                        _i = i
                        _j = j
                        di = direction[0]
                        dj = direction[1]

                        buf = []

                        while True:
                            _i = _i + di
                            _j = _j + dj

                            current_pos = board[_i][_j]

                            if current_pos == "?":
                                # invalid position: no flank to be found along this path
                                break

                            elif current_pos == ".":
                                # blank position: no flank to be found along this path
                                break

                            elif current_pos == color and len(buf) == 0:
                                # neighbouring already controlled coin: no flank to be found here
                                break

                            elif current_pos == target_color:
                                # storing coordinates of current coin
                                #   as it will be flipped if we find a valid
                                #   flank along this path
                                buf.append((_i, _j))
                                continue

                            else:
                                # flank detected: this is a valid move;
                                found_flank, found_move = True, True
                                flanked_coins.extend(buf)
                                break

                    if not found_flank:
                        # moving to next position on the board
                        continue

                    # cloning board
                    updated_board = deepcopy(board)

                    # effectively making the move
                    updated_board[i][j] = color

                    # flipping flanked coins
                    for coin in flanked_coins:
                        updated_board[coin[0]][coin[1]] = color

                    _, score = self.minimax(
                                            updated_board, 
                                            not should_maximize, 
                                            depth_level + 1, 
                                            max_depth, 
                                            alpha, 
                                            beta
                                            )

                    # applying alpha-beta pruning
                    if should_maximize:
                        if score >= beta: return ((i, j), score)
                        alpha = max(alpha, score)
                    else:
                        if score <= alpha: return ((i, j), score)
                        beta = min(beta, score)

                    possible_moves.append(((i, j), score))

        if not found_move:
            # this is a leaf node (terminal state),
            #   so we must stop the recursion and evaluate
            score = self.evaluate_board_configuration(board)
            return (None, score)

        global total_branches
        global visited_nodes
        total_branches += len(possible_moves)
        visited_nodes += 1

        # finding best move based on current node's strategy (to maximize or to minimize)
        coord, best_score = None, float("-inf") if should_maximize else float("inf")
        for c, score in possible_moves:
            if ((should_maximize and score > best_score) or (not should_maximize and score < best_score)):
                coord = c
                best_score = score

        return (coord, best_score)


    def play(self, board_wrapper):
        """ 
            Method implementing the expected interface for a player

            @params:
                board_wrapper: an object wrapping the current board configuration;

            @returns:
                The chosen move.
        """
        # measuring time we're taking to make the play
        #   by the tournament rules,
        #   we have at most 3 seconds to make a move.
        start = time.time()

        if not self.first_turn:
            # incrementing turn-count by two,
            #   to keep up-to-date with the 
            #   turn our opponent just played
            self.turn += 2
        else:
            self.first_turn = False
            if self.turn == 1:
                # returning default first move
                #   if we're the first player to play
                return Move(4, 3)

        # unwrapping board configuration object
        board = board_wrapper.board

        depthness = 4
        while True:
            # executing minimax with alpha-beta pruning using an iterative DFS
            elapsed_in_iteration = time.time()
            move, score = self.minimax(board, True, 0, depthness, float("-inf"), float("inf"))
            elapsed_in_iteration = time.time() - elapsed_in_iteration
            elapsed = time.time() - start

            print "Depthness {} took {} to run".format(str(depthness), str(elapsed_in_iteration))

            if elapsed + (elapsed_in_iteration * 8) >= 2.5:
                break

            depthness += 1

        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print "~~~~ Reached depthness {} & took {} to play ~~~~".format(str(depthness), str(time.time() - start))
        print "~~~~ Branching factor this iteration was {} ~~~~".format(str(total_branches / visited_nodes))
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"


        return Move(move[0], move[1])
