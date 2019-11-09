class NoobNoob:
    """ 
        Simulates an inteligent player
    """

    import time
    import copy

    from models.move import Move

    def __init__(self, color):
        """
            Constructor.
        """
        self.DEBUG_FLAG = False

        self.color = color
        self.opponent_color = "o" if self.color == "@" else "@"

        self.turn = 1 if self.color == "@" else 2
        self.first_turn = True

        self.CORNERS = [
            (1, 1),   # north-western corner
            (1, 8),   # north-eastern corner
            (8, 8),   # south-eastern corner
            (8, 1)    # south-western corner
        ]

        self.CORNER_NEIGHBOURS = [
            (2, 1), (2, 2), (1, 2),
            (1, 7), (2, 7), (2, 8),
            (7, 1), (7, 2), (8, 2),
            (8, 7), (7, 7), (7, 8)
        ]

        self.DIRECTIONS = [
            (-1, -1), # north-west
            (-1, 0),  # north
            (-1, 1),  # north-east
            (0, 1),   # east
            (1, 1),   # south-east
            (1, 0),   # south
            (1, -1),  # south-west
            (0, -1)   # west
        ]


    def evaluate_board_configuration(self, board):
        """
            Evaluates the current board configuration based on the linear combination
                of the heuristics described below. The intuition for the heuristics was 
                taken from the articles below:
                # https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA/Final_Paper.pdf
                # https://pdfs.semanticscholar.org/235f/b5f2ebae93e33e2bf7038bb37a690fa9390e.pdf


            The heuristics under employment are:

            * Disc Parity
            * Corners
            * Stability
            * Mobility

            @params:
                board: the board configuration to be evaluated;

            @returns:
                A score.
        """
        # counters for disc parity heuristic
        max_discs, min_discs = 0, 0

        # counters for potential mobility heuristic
        max_frontier_discs = set()
        min_frontier_discs = set()
        max_potential_moves = set()
        min_potential_moves = set()

        # counters for edge heuristic
        max_edge_discs, min_edge_discs = 0, 0

        # iterating over the board and evaluating current configuration of discs:
        for i in range(1, 9):
            min_discs_in_row = 0
            max_discs_in_row = 0

            for j in range(1, 9):
                disc = board[i][j]
                if disc == self.color:
                    # disc parity
                    max_discs_in_row += 1
                    max_discs += 1

                    if i == 1 or i == 8 or j == 1 or j == 8:
                        # this is an edge, so this disc is semi-stable
                        max_edge_discs += 1

                    # potential mobility
                    for direction in self.DIRECTIONS:
                        _i, _j = i + direction[0], j + direction[1]
                        if board[_i][_j] == ".":
                            # adding to potential moves set
                            if (_i, _j) not in min_potential_moves:
                                min_potential_moves.add((_i, _j))
                            # adding to frontier discs set
                            if (i, j) not in min_frontier_discs:
                                min_frontier_discs.add((i, j))

                elif disc == self.opponent_color:
                    # disc parity
                    min_discs_in_row += 1
                    min_discs += 1

                    if i == 1 or i == 8 or j == 1 or j == 8:
                        # this is an edge, so this disc is semi-stable
                        min_edge_discs += 1

                    # potential mobility
                    for direction in self.DIRECTIONS:
                        _i, _j = i + direction[0], j + direction[1]
                        if board[_i][_j] == "." and (_i, _j):
                            # adding to potential moves set
                            if (_j, _j) not in max_potential_moves:
                                max_potential_moves.add((_i, _j))
                            # adding to frontier discs set
                            if (i, j) not in max_frontier_discs:
                                max_frontier_discs.add((i, j))

        # calculating disc parity score
        disc_parity_score = (100 * (float(max_discs - min_discs) / float(max_discs + min_discs)))

        mobility_score = 0
        max_potential_mobility = len(max_potential_moves) + len(max_frontier_discs)
        min_potential_mobility = len(min_potential_moves) + len(min_frontier_discs)
        # calculating potential mobility score
        if max_potential_mobility + min_potential_mobility != 0:
            mobility_score = (100 * 
                             (float(max_potential_mobility - min_potential_mobility) / 
                             float(max_potential_mobility + min_potential_mobility)))

        edge_score = 0
        # calculating edge score
        if max_edge_discs + min_edge_discs != 0:
            edge_score = (100 *
                         (float(max_edge_discs - min_edge_discs) /
                         float(max_edge_discs + min_edge_discs)))

        # counters for corners heuristics
        max_corners_captured = 0
        min_corners_captured = 0
        max_potential_corners = 0
        min_potential_corners = 0
        max_potential_corners_considered = set()
        min_potential_corners_considered = set()

        # evaluating corners:
        for corner_coords in self.CORNERS:
            corner_color = board[corner_coords[0]][corner_coords[1]]

            if corner_color == self.color:
                # corner captured by max player
                max_corners_captured += 1
                continue

            if corner_color == self.opponent_color:
                # corner captured by min player
                min_corners_captured += 1
                continue

            # corner uncaptured
            # evaluating potential corners:
            for direction in self.DIRECTIONS:
                _i, _j = corner_coords[0], corner_coords[1]
                di, dj = direction[0], direction[1]

                _i += di
                _j += dj

                current_pos = board[_i][_j]

                if current_pos == "?" or current_pos == ".":
                    continue

                if current_pos == self.opponent_color:
                    if corner_coords not in max_potential_corners_considered:
                        max_potential_corners_considered.add(corner_coords)
                        max_potential_corners += 1

                if current_pos == self.color:
                    if corner_coords not in min_potential_corners_considered:
                        min_potential_corners_considered.add(corner_coords)
                        min_potential_corners += 1

        captured_corners_score = 0
        # calculating captured corner scores
        if max_corners_captured + min_corners_captured != 0:
            captured_corners_score = (100 * 
                                     (float(max_corners_captured - min_corners_captured) / 
                                     float(max_corners_captured + min_corners_captured)))

        potential_corners_score = 0
        # calculating potential corner scores
        if max_potential_corners + min_potential_corners != 0:
            potential_corners_score = (100 *
                                      (float(max_potential_corners - min_potential_corners) /
                                      float(max_potential_corners + min_potential_corners)))

        return disc_parity_score + mobility_score + 10 * (captured_corners_score + potential_corners_score) + edge_score


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

        found_move = False
        possible_moves = []

        for i in range(1, 9):
            for j in range(1, 9):
                if board[i][j] == ".":

                    flanked_discs = []
                    found_flank = False

                    for direction in self.DIRECTIONS:
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
                                # neighbouring already controlled disc: no flank to be found here
                                break

                            elif current_pos == target_color:
                                # storing coordinates of current disc
                                #   as it will be flipped if we find a valid
                                #   flank along this path
                                buf.append((_i, _j))
                                continue

                            else:
                                # flank detected: this is a valid move;
                                found_flank, found_move = True, True
                                flanked_discs.extend(buf)
                                break

                    if not found_flank:
                        # moving to next position on the board
                        continue

                    # cloning board
                    # updated_board = self.deepcopy(board)
                    updated_board = self.copy.deepcopy(board)

                    # effectively making the move
                    updated_board[i][j] = color

                    # flipping flanked discs
                    for disc in flanked_discs:
                        updated_board[disc[0]][disc[1]] = color

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
        start = self.time.time()

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
                return self.Move(4, 3)

        # unwrapping board configuration object
        board = board_wrapper.board

        corner_checks = self.time.time()
        # we always capture a corner if we can
        if self.turn > 20:
            for corner_coords in self.CORNERS:
                corner_color = board[corner_coords[0]][corner_coords[1]]
                if corner_color == ".":
                    for direction in self.DIRECTIONS:
                        _i, _j = corner_coords[0], corner_coords[1]
                        di, dj = direction[0], direction[1]

                        _i += di
                        _j += dj
                        current_pos = board[_i][_j]

                        if current_pos != self.opponent_color:
                            continue

                        found_flank = False
                        while True:
                            _i += di
                            _j += dj
                            current_pos = board[_i][_j]

                            if current_pos == "?" or current_pos == ".":
                                # no flank found along this path
                                break

                            if current_pos == self.color:
                                # found flank, we can capture this corner!
                                return self.Move(corner_coords[0], corner_coords[1])
        corner_checks = self.time.time() - corner_checks

        depthness = 4
        while True:
            # executing minimax with alpha-beta pruning using an iterative DFS
            elapsed_in_iteration = self.time.time()
            move, score = self.minimax(board, True, 0, depthness, float("-inf"), float("inf"))
            elapsed_in_iteration = self.time.time() - elapsed_in_iteration
            elapsed = self.time.time() - start

            if elapsed + (elapsed_in_iteration * 8) >= 2.5:
                break

            depthness += 1

        if self.DEBUG_FLAG:
            print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            print "Turn {}".format(str(self.turn))
            print "Reached depthness {}".format(str(depthness))
            print "Took {} to play".format(str(self.time.time() - start))
            print "Took {} to check corners".format(str(corner_checks))
            print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

        return self.Move(move[0], move[1])
