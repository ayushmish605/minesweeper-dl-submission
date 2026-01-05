"""
This is my reference Logic Bot for Minesweeper.

I use it in two ways:
- as a baseline agent for comparison
- as a "teacher" to generate realistic training states for Task 1

Interface (shared with NN agents):
- input: visible board from `game.get_visible_board()`
- action: `(row, col)` for clicks (and an optional flag action dict for UI/replay)
- step API: `play_step()` -> `(result, action)` where result is `"Cont"|"Win"|"Lost"|"Done"`
"""

import random
from typing import Tuple, Set, Dict, Optional, List, Any
from .game import MinesweeperGame, GameState


class LogicBot:
    """
    This bot is my straight implementation of the inference rules described in the assignment.

    High-level loop:
    - I maintain `cells_remaining`, `inferred_safe`, `inferred_mine`, and a `clue_number` map.
    - If I know a safe cell, I click it; otherwise I fall back to a random unrevealed cell.
    - After each click, I update clues and repeatedly apply the two standard Minesweeper
      inference rules until nothing new is discovered.
    """
    
    def __init__(self, game: MinesweeperGame, seed: Optional[int] = None):
        """
        Initialize the logic bot.
        
        Args:
            game: The MinesweeperGame instance to play
            seed: Random seed for reproducibility
        """
        self.game = game
        if seed is not None:
            random.seed(seed)
        
        # Initialize the internal sets for inference.
        self.cells_remaining: Set[Tuple[int, int]] = set()
        self.inferred_safe: Set[Tuple[int, int]] = set()
        self.inferred_mine: Set[Tuple[int, int]] = set()
        self.clue_number: Dict[Tuple[int, int], int] = {}  # Hashmap for clue numbers
        # I track flags we "placed" so we don't repeat flag actions.
        self._flags_placed: Set[Tuple[int, int]] = set()
        
        # Initialize cells_remaining with all cells.
        for row in range(game.height):
            for col in range(game.width):
                self.cells_remaining.add((row, col))

        # Then I ensure sets are consistent with any pre-revealed board state.
        self._sync_sets_with_visible()

    def _next_flag_action(self) -> Optional[Tuple[int, int]]:
        """
        If we have inferred mines that are still unrevealed and not yet flagged, return one.
        """
        self._sync_sets_with_visible()
        for (r, c) in list(self.inferred_mine):
            # Only flag unrevealed cells (visible board still 'E')
            try:
                if self.game.board[r][c] != "E":
                    continue
            except Exception:
                continue
            if (r, c) in self._flags_placed:
                continue
            return (r, c)
        return None

    def _sync_sets_with_visible(self) -> None:
        """
        Synchronize internal sets with the current visible board.

        The game can reveal multiple cells at once (DFS cascade). If we don't prune those
        from `cells_remaining` / `inferred_safe`, the bot may repeatedly "click" already
        revealed cells and stop making progress.
        """
        visible = self.game.get_visible_board()
        for r in range(self.game.height):
            for c in range(self.game.width):
                if visible[r][c] != "E":
                    self.cells_remaining.discard((r, c))
                    self.inferred_safe.discard((r, c))
                    # Once revealed, it's no longer an "unrevealed mine candidate".
                    # BUT: in "continue after mine triggers", mines can become visible as "M".
                    # Those should still be treated as known mines for future inference.
                    if visible[r][c] != "M":
                        self.inferred_mine.discard((r, c))
    
    def get_game_state(self) -> List[List[str]]:
        """
        Get the current visible game state.
        
        Returns:
            Visible board as list of lists (same format neural networks will receive)
        """
        return self.game.get_visible_board()
    
    def select_action(self) -> Tuple[int, int]:
        """
        Select the next action to take.
        
        Returns:
            (row, col) tuple - same format neural networks will return
        """
        # Keep internal sets in sync before selecting.
        self._sync_sets_with_visible()

        # If inferred_safe is not empty, pick a still-unrevealed cell from it
        while len(self.inferred_safe) > 0:
            cell = self.inferred_safe.pop()
            r, c = cell
            # Only click unrevealed cells; skip anything already opened by cascade.
            if self.game.board[r][c] == "E":
                return cell
        
        # Otherwise, select a cell from cells_remaining at random
        if len(self.cells_remaining) == 0:
            # No cells left (shouldn't happen in normal play)
            return None
        
        # Ensure random fallback also only picks unrevealed cells (extra safety).
        candidates = [(r, c) for (r, c) in self.cells_remaining if self.game.board[r][c] == "E"]
        if not candidates:
            return None
        return random.choice(candidates)

    def select_click_action(self) -> Optional[Dict[str, Any]]:
        """
        Select a CLICK action, tagged with whether it was deterministic (inferred safe)
        or random (fallback).

        Returns:
            {"type": "deterministic"|"random", "pos": [row, col]} or None
        """
        # Keep internal sets in sync before selecting.
        self._sync_sets_with_visible()

        # Deterministic: inferred safe cells
        while len(self.inferred_safe) > 0:
            cell = self.inferred_safe.pop()
            r, c = cell
            if self.game.board[r][c] == "E":
                return {"type": "deterministic", "pos": [int(r), int(c)]}

        # Random fallback
        if len(self.cells_remaining) == 0:
            return None
        candidates = [(r, c) for (r, c) in self.cells_remaining if self.game.board[r][c] == "E"]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        return {"type": "random", "pos": [int(r), int(c)]}
    
    def make_inferences(self) -> bool:
        """
        Make logical inferences about safe cells and mines.

        Sound inference rules:
        - If (clue - #neighbors inferred mines) == (# unrevealed neighbors), all unrevealed neighbors are mines.
        - If (clue == #neighbors inferred mines), all other unrevealed neighbors are safe.
        
        Returns:
            True if new inferences were made, False otherwise
        """
        new_inferences = False
        
        visible = self.game.get_visible_board()

        # For each cell with a revealed clue
        for (row, col), clue in self.clue_number.items():
            neighbors = self._get_neighbors(row, col)
            
            # Count neighbors inferred to be mines
            # IMPORTANT: when "continue after mine triggers" is enabled, mines may become
            # Visible as "M". Those should count as known mines too, otherwise inference
            # Becomes unsound after the first triggered mine.
            neighbors_inferred_mines = sum(
                1
                for nr, nc in neighbors
                if (nr, nc) in self.inferred_mine or (visible[nr][nc] == "M")
            )
            
            # Count unrevealed neighbors (not inferred as mines)
            unrevealed_neighbors = [(nr, nc) for nr, nc in neighbors 
                                   if self.game.board[nr][nc] == 'E' and 
                                   (nr, nc) not in self.inferred_mine]
            
            # Rule 1 (mines): remaining mines around this clue must occupy all unrevealed neighbors.
            if len(unrevealed_neighbors) > 0:
                if clue - neighbors_inferred_mines == len(unrevealed_neighbors):
                    for nr, nc in unrevealed_neighbors:
                        if (nr, nc) not in self.inferred_mine:
                            self.inferred_mine.add((nr, nc))
                            self.cells_remaining.discard((nr, nc))
                            new_inferences = True

            # Rule 2 (safe): if all mines around this clue are already accounted for, the rest are safe.
            if len(unrevealed_neighbors) > 0:
                if clue == neighbors_inferred_mines:
                    for nr, nc in unrevealed_neighbors:
                        if (nr, nc) not in self.inferred_safe:
                            self.inferred_safe.add((nr, nc))
                            # NOTE: do not discard inferred_safe from cells_remaining; random fallback
                            # I should still have a valid pool even if inferred_safe gets exhausted by
                            # Cascade reveals.
                            new_inferences = True
        
        return new_inferences
    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all valid neighboring cell positions."""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.game.height and 0 <= nc < self.game.width:
                    neighbors.append((nr, nc))
        return neighbors
    
    def _update_clue_numbers(self):
        """Update clue_number hashmap for all revealed cells."""
        visible = self.game.get_visible_board()
        for r in range(self.game.height):
            for c in range(self.game.width):
                cell_value = visible[r][c]
                # If cell has a clue number (0-8), save it
                if cell_value not in ['E', 'B', 'M']:
                    try:
                        clue = int(cell_value)
                        if (r, c) not in self.clue_number:
                            self.clue_number[(r, c)] = clue
                    except ValueError:
                        pass
    
    def play_step(self) -> Tuple[str, Optional[Any]]:
        """
        Play a single step (one action).
        
        Returns:
            Tuple of (result, action_taken)
            result: "Cont", "Win", "Lost", or "Done"
            action_taken:
              - click: (row, col)
              - flag: {"type": "flag", "pos": [row, col]}
              - None
        """
        # Normally, I stop once the game is not in progress.
        
        # However, this project supports an optional "continue after mine" mode where
        # The underlying game state becomes LOST after a mine trigger, but we still want
        # I do this to keep selecting actions until we either finish (DONE/WON) or run out of moves.
        gs = self.game.get_game_state()
        allow = bool(getattr(self.game, "allow_mine_triggers", False))
        if gs != GameState.PROG:
            if not (allow and gs == GameState.LOST):
                return ("Done", None)

        # If we've logically inferred mines, place flags first.
        flag_target = self._next_flag_action()
        if flag_target is not None:
            r, c = flag_target
            self._flags_placed.add(flag_target)
            # Do not change the game board; flags are a UI / replay artifact.
            return ("Cont", {"type": "flag", "pos": [int(r), int(c)]})
        
        # Select click action
        action = self.select_click_action()
        if not action:
            return ("Done", None)
        
        try:
            row, col = int(action["pos"][0]), int(action["pos"][1])
        except Exception:
            return ("Done", None)

        self.cells_remaining.discard((row, col))
        
        # Open the selected cell
        buttons_clear = set()
        result = self.game.player_clicks(row, col, buttons_clear)

        # Prune any cascade-revealed cells so we don't re-click them later.
        self._sync_sets_with_visible()
        
        if result == "Lost":
            return ("Lost", action)
        
        # I update clue_number hashmap
        self._update_clue_numbers()
        
        # I make inferences iteratively until no new inferences
        while True:
            made_inference = self.make_inferences()
            if not made_inference:
                break
        
        if result == "Win":
            return ("Win", action)
        
        return ("Cont", action)
    
    def play(self, allow_mine_triggers: bool = False) -> dict:
        """
        Play a complete game using the logic bot.
        
        Args:
            allow_mine_triggers: If True, continue after triggering a mine (not used in spec)
        
        Returns:
            Dictionary with:
            - cells_opened: Number of cells successfully opened or revealed as safe
            - mines_triggered: Number of mines triggered
            - game_won: Whether the game was won
            - steps: Number of steps taken
        """
        steps = 0
        
        # Keep the game object and this function in sync (GUI may also set game.allow_mine_triggers).
        try:
            setattr(self.game, "allow_mine_triggers", bool(allow_mine_triggers))
        except Exception:
            pass

        # Loop until end of game.
        # With allow_mine_triggers=True, I keep going even if the state becomes LOST.
        while True:
            gs = self.game.get_game_state()
            if gs == GameState.PROG:
                pass
            elif allow_mine_triggers and gs == GameState.LOST:
                pass
            else:
                break
            steps += 1
            result, action = self.play_step()
            
            if result == "Lost":
                if not allow_mine_triggers:
                    break
                continue
            elif result == "Win" or result == "Done":
                break
        
        stats = self.game.get_statistics()
        return {
            'cells_opened': stats['cells_opened'],
            'mines_triggered': stats['mines_triggered'],
            'game_won': stats['game_won'],
            'steps': steps
        }
