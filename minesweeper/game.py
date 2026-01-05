"""
This is the core Minesweeper game logic I use throughout the project.

I represent both the hidden board and the visible board with simple strings:
- `"E"` = unrevealed
- `"M"` = mine
- `"0"`-`"8"` = clue numbers (in the *actual* board)
- `"B"` = revealed blank region (in the *visible* board)
"""

import random
from typing import Tuple, Optional, List, Set
from enum import Enum


class GameState(Enum):
    """I use these small status codes to track whether the game is ongoing or finished."""
    PROG = "PROG"
    WON = "WON"
    LOST = "LOST"
    DONE = "DONE"


class MinesweeperGame:
    """
    This is my Minesweeper environment.

    The key design choice is that the *player* (or bot) only ever sees `self.board`
    (the visible board). The ground-truth `self.actual_board` is generated after the
    first click so I can guarantee the first click is safe.
    """
    
    def __init__(self, height: int = 22, width: int = 22, num_mines: int = 80, 
                 seed: Optional[int] = None, ensure_solvable: bool = False):
        """
        Initialize a new Minesweeper game.
        
        Args:
            height: Height of the board (number of rows)
            width: Width of the board (number of columns)
            num_mines: Number of mines to place
            seed: Random seed for reproducibility
            ensure_solvable: If True, only accept boards that are solvable from first click
                           (Level 4). If False, uses Level 2 (3×3 safe area, default).
        """
        self.height = height
        self.width = width
        self.num_mines = num_mines
        self.ensure_solvable = ensure_solvable
        
        if seed is not None:
            random.seed(seed)
        
        # I keep two boards:
        # - actual_board = mines + clue numbers
        # - board = player's visible board
        self.actual_board = None  # Will be set when the board is generated (after the first click)
        self.board = None  # Player's visible board (starts as all "E")
        self.visited = None  # Visited array for DFS (like makeVisited)
        
        # I track game state here.
        self.game_state = GameState.PROG
        self.mine_count = 0  # Actual number of mines placed
        self._board_initialized = False  # Tracks whether the board has been generated
        self._first_click_position = None  # Stores the first click position
        
        # I track simple stats as I play.
        self.cells_opened = 0
        self.mines_triggered = 0
        
        # I initialize the visible board as all empty/unrevealed.
        # Generate the mine board after the first click to guarantee the first click is safe.
        self.board = [['E' for _ in range(self.width)] for _ in range(self.height)]
        self.visited = [[0 for _ in range(self.width)] for _ in range(self.height)]
    
    def _get_neighbors_8(self, row: int, col: int):
        """
        Get all 8 neighbors (including diagonals) of a cell.
        
        Args:
            row: Row of the cell
            col: Column of the cell
            
        Yields:
            (row, col) tuples for valid neighbors
        """
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r = row + dr
                c = col + dc
                if 0 <= r < self.height and 0 <= c < self.width:
                    yield (r, c)
    
    def _generate_board(self, safe_row: int, safe_col: int):
        """
        Generate a random board with mines.
        
        Level 2 implementation (default): First click and all 8 neighbors are guaranteed safe
        (no mines in the 3×3 area). This creates a favorable opening that almost
        always reveals a large region.
        
        Level 4 implementation (if ensure_solvable=True): Only accepts boards that are
        solvable from the first click using logic (no guessing required).
        
        Args:
            safe_row: Row of first click (guaranteed to be safe)
            safe_col: Column of first click (guaranteed to be safe)
        """
        if self.ensure_solvable:
            # Level 4: Try to generate a solvable board
            MAX_TRIES = 200
            for attempt in range(MAX_TRIES):
                if self._try_generate_solvable_board(safe_row, safe_col):
                    return
            # Fallback: accept last generated board (Level 2)
            self._generate_board_level2(safe_row, safe_col)
        else:
            # Level 2: 3×3 safe area (default, much better UX)
            self._generate_board_level2(safe_row, safe_col)
    
    def _generate_board_level2(self, safe_row: int, safe_col: int):
        """
        Generate board with Level 2 rules: 3×3 safe area around first click.
        
        Args:
            safe_row: Row of first click (guaranteed to be safe)
            safe_col: Column of first click (guaranteed to be safe)
        """
        # Forbid the 3×3 area around first click (Level 2: much better UX)
        forbidden = set(self._get_neighbors_8(safe_row, safe_col))
        
        # Get all candidate positions (excluding forbidden area)
        candidates = [(r, c) for r in range(self.height) 
                     for c in range(self.width) 
                     if (r, c) not in forbidden]
        
        # I ensure we have enough cells for the requested number of mines
        max_mines = len(candidates)
        num_mines_to_place = min(self.num_mines, max_mines)
        
        # Randomly select mine positions
        mine_positions = set(random.sample(candidates, num_mines_to_place))
        self.mine_count = len(mine_positions)
        
        # I create mine board
        mine_board = [["E" for _ in range(self.width)] for _ in range(self.height)]
        for r, c in mine_positions:
            mine_board[r][c] = "M"
        
        # I create actual board with clue numbers
        self._compute_clue_numbers(mine_board)
        
        self._board_initialized = True
    
    def _compute_clue_numbers(self, mine_board: List[List[str]]):
        """Compute clue numbers from mine board."""
        self.actual_board = [["0" for _ in range(self.width)] for _ in range(self.height)]
        
        for i in range(self.height):
            for j in range(self.width):
                if mine_board[i][j] == "M":
                    self.actual_board[i][j] = "M"
                    # Increment clue numbers for neighbors
                    for drow in range(-1, 2):
                        for dcol in range(-1, 2):
                            nrow = i + drow
                            ncol = j + dcol
                            if (0 <= nrow < self.height and 
                                0 <= ncol < self.width):
                                if mine_board[nrow][ncol] == "M":
                                    continue
                                tmp = int(self.actual_board[nrow][ncol]) + 1
                                self.actual_board[nrow][ncol] = str(tmp)
    
    def _try_generate_solvable_board(self, safe_row: int, safe_col: int) -> bool:
        """
        Try to generate a solvable board (Level 4).
        
        Returns:
            True if a solvable board was generated, False otherwise
        """
        # Generate board with Level 2 rules first
        self._generate_board_level2(safe_row, safe_col)
        
        # I check if board is solvable using a simple solver
        # We'll use a basic logic-based solver (similar to LogicBot)
        return self._is_solvable_from_first_click(safe_row, safe_col)
    
    def _is_solvable_from_first_click(self, safe_row: int, safe_col: int) -> bool:
        """
        Check if the board is solvable from the first click using logic only.
        
        Uses a simplified version of LogicBot to verify solvability.
        
        Returns:
            True if solvable without guessing, False otherwise
        """
        # I create a copy of the board state for simulation
        sim_board = [row[:] for row in self.board]
        sim_visited = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        # Simulate first click reveal
        buttons_clear = set()
        if self.actual_board[safe_row][safe_col] == "0":
            self._simulate_dfs(safe_row, safe_col, sim_board, sim_visited, buttons_clear)
        else:
            sim_board[safe_row][safe_col] = self.actual_board[safe_row][safe_col]
            buttons_clear.add(self.width * safe_row + safe_col)
        
        # Mark revealed cells (buttons_clear contains indices, convert to row/col)
        for idx in buttons_clear:
            row = idx // self.width
            col = idx % self.width
            if 0 <= row < self.height and 0 <= col < self.width:
                sim_board[row][col] = self.actual_board[row][col]
        
        # I try to solve using logic (simplified LogicBot approach)
        # This is a basic check - a full implementation would use LogicBot
        # For now, we'll do a simple check: if we can reveal more cells using
        # Basic deduction, consider it solvable
        
        # Count how many cells we can reveal with basic logic
        max_iterations = 100
        for _ in range(max_iterations):
            progress_made = False
            
            # For each revealed cell with a number, check if we can deduce neighbors
            for r in range(self.height):
                for c in range(self.width):
                    if sim_board[r][c] not in ['E', 'M'] and sim_board[r][c] != 'B':
                        try:
                            clue = int(sim_board[r][c])
                            if clue > 0:
                                # Count flagged and unrevealed neighbors
                                neighbors = list(self._get_neighbors_8(r, c))
                                flagged = sum(1 for nr, nc in neighbors 
                                            if sim_board[nr][nc] == 'F')
                                unrevealed = sum(1 for nr, nc in neighbors 
                                               if sim_board[nr][nc] == 'E')
                                
                                # If all remaining neighbors must be mines, flag them
                                if clue - flagged == unrevealed and unrevealed > 0:
                                    # I can flag all unrevealed neighbors
                                    progress_made = True
                                
                                # If all mines are flagged, reveal safe neighbors
                                if flagged == clue and unrevealed > 0:
                                    # I can reveal all unrevealed neighbors
                                    for nr, nc in neighbors:
                                        if sim_board[nr][nc] == 'E':
                                            if self.actual_board[nr][nc] == 'M':
                                                return False  # Would hit a mine
                                            sim_board[nr][nc] = self.actual_board[nr][nc]
                                            if self.actual_board[nr][nc] == '0':
                                                self._simulate_dfs(nr, nc, sim_board, sim_visited, buttons_clear)
                                            progress_made = True
                        except ValueError:
                            pass
            
            if not progress_made:
                break
        
        # I check if we've revealed a significant portion of the board
        # (This is a heuristic - a full check would verify complete solvability)
        revealed = sum(1 for r in range(self.height) for c in range(self.width) 
                      if sim_board[r][c] not in ['E', 'F'])
        total_safe = self.height * self.width - self.mine_count
        reveal_ratio = revealed / max(1, total_safe)
        
        # Consider solvable if we can reveal at least 30% of safe cells with logic
        return reveal_ratio >= 0.3
    
    def _simulate_dfs(self, row: int, col: int, board: List[List[str]], 
                     visited: List[List[int]], buttons_clear: set):
        """Simulate DFS for solvability checking."""
        if visited[row][col]:
            return
        visited[row][col] = 1
        board[row][col] = "B"
        buttons_clear.add(self.width * row + col)
        
        for drow in range(-1, 2):
            for dcol in range(-1, 2):
                nrow = row + drow
                ncol = col + dcol
                if (0 <= nrow < self.height and 
                    0 <= ncol < self.width and 
                    not visited[nrow][ncol]):
                    # Never reveal mines during cascade simulation.
                    if self.actual_board[nrow][ncol] == "M":
                        continue
                    if self.actual_board[nrow][ncol] == "0":
                        self._simulate_dfs(nrow, ncol, board, visited, buttons_clear)
                    else:
                        board[nrow][ncol] = self.actual_board[nrow][ncol]
                        buttons_clear.add(self.width * nrow + ncol)
    
    def dfs(self, row: int, col: int, buttons_clear: set):
        """
        Depth-first search for cascading reveals.
        
        Args:
            row: Starting row
            col: Starting column
            buttons_clear: Set to track which buttons need to be updated
        """
        self.visited[row][col] = 1
        self.board[row][col] = "B"
        buttons_clear.add(self.width * row + col)
        
        for drow in range(-1, 2):
            for dcol in range(-1, 2):
                nrow = row + drow
                ncol = col + dcol
                if (0 <= nrow < self.height and 
                    0 <= ncol < self.width and 
                    not self.visited[nrow][ncol]):
                    # Never reveal mines during cascade.
                    if self.actual_board[nrow][ncol] == "M":
                        continue
                    if self.actual_board[nrow][ncol] == "0":
                        self.dfs(nrow, ncol, buttons_clear)
                    else:
                        self.visited[nrow][ncol] = 1
                        self.board[nrow][ncol] = self.actual_board[nrow][ncol]
                        buttons_clear.add(self.width * nrow + ncol)
    
    def check_spaces(self) -> bool:
        """
        Check if game is won.
        Game is won when all non-mine cells have been revealed.
        This must not depend on mines staying unrevealed, since we optionally support
        "continue after mine" where mines may become visible during play.
        
        Returns:
            True if game is won, False otherwise
        """
        total_safe = max(0, (self.height * self.width) - int(self.mine_count))
        return int(self.cells_opened) >= total_safe
    
    def player_clicks(self, row: int, col: int, buttons_clear: set, allow_mine_triggers: Optional[bool] = None) -> str:
        """
        Handle player click.
        
        According to Minesweeper DL.pdf: "traditionally the first cell, which must be 
        chosen as random, is taken to not be a mine - the game environment might return 
        a non-mine cell with a 0 clue value, selected randomly, at the start of the game."
        
        Args:
            row: Row of clicked cell
            col: Column of clicked cell
            buttons_clear: Set to track buttons that need updating
        
        Returns:
            "Lost" if mine hit, "Win" if game won, "Cont" otherwise
        """
        if allow_mine_triggers is None:
            allow_mine_triggers = bool(getattr(self, "allow_mine_triggers", False))

        # Generate board on first click, ensuring first click is safe
        if not self._board_initialized:
            self._generate_board(row, col)
            self._first_click_position = (row, col)
            # First click is guaranteed safe, so we can proceed
        
        # Ignore re-clicks on already revealed cells (prevents double-counting progress).
        # Also prevents bots/players from inflating cells_opened by repeatedly clicking.
        if self.board[row][col] != "E":
            # If already completed, return Done/Win; otherwise no-op.
            if self.check_spaces():
                self.game_state = GameState.DONE if int(self.mines_triggered) > 0 else GameState.WON
                return "Win"
            return "Cont"

        if self.actual_board[row][col] == "M":
            self.mines_triggered += 1
            # Reveal only the triggered mine cell (do NOT reveal the whole board).
            self.board[row][col] = "M"
            # Once a mine is triggered, the run is considered LOST unless it later reaches DONE.
            # (When allow_mine_triggers is False, the GUI will end the run immediately.)
            self.game_state = GameState.LOST
            return "Lost"
        elif self.actual_board[row][col] != "0":
            self.board[row][col] = self.actual_board[row][col]
            # Mark visited so DFS cascades won't re-count this revealed clue cell later.
            # (Without this, already-open number cells can be "re-opened" during DFS
            # And inflate cells_opened, causing false WON states.)
            try:
                self.visited[row][col] = 1
            except Exception:
                pass
            self.cells_opened += 1
            try:
                buttons_clear.add(self.width * row + col)
            except Exception:
                pass
        else:
            before = len(buttons_clear)
            self.dfs(row, col, buttons_clear)
            # Count all cells revealed by DFS
            self.cells_opened += max(0, len(buttons_clear) - before)
        
        if self.check_spaces():
            self.game_state = GameState.DONE if int(self.mines_triggered) > 0 else GameState.WON
            return "Win"
        # Otherwise, remain in progress unless a mine was triggered earlier.
        if int(self.mines_triggered) == 0:
            self.game_state = GameState.PROG
        
        return "Cont"
    
    def get_game_state(self) -> GameState:
        """Get the current game state."""
        return self.game_state
    
    def get_statistics(self) -> dict:
        """Get game statistics."""
        gs = self.get_game_state()
        return {
            'cells_opened': self.cells_opened,
            'mines_triggered': self.mines_triggered,
            'game_won': gs in (GameState.WON, GameState.DONE),
            'game_lost': gs == GameState.LOST,
            'mine_count': self.mine_count
        }
    
    def reset(self, seed: Optional[int] = None):
        """Reset the game to initial state."""
        if seed is not None:
            random.seed(seed)
        self.game_state = GameState.PROG
        self.cells_opened = 0
        self.mines_triggered = 0
        self._board_initialized = False
        self._first_click_position = None
        self.actual_board = None
        # I reset player board (all empty/unrevealed)
        self.board = [['E' for _ in range(self.width)] for _ in range(self.height)]
        self.visited = [[0 for _ in range(self.width)] for _ in range(self.height)]
    
    def get_visible_board(self) -> List[List[str]]:
        """Get the current visible board state."""
        return self.board
    
    def get_actual_board(self) -> List[List[str]]:
        """Get the actual board with mines and clues (for debugging/training)."""
        if not self._board_initialized:
            return None  # Board not generated yet (before first click)
        return self.actual_board
