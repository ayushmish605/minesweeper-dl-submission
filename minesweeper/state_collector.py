"""
This is the state collector I use to save Minesweeper runs for training/evaluation.

The main idea is: I log every action, but I only snapshot the full visible board
periodically ("keyframes") so the JSON stays reasonably small.
"""

import os
import json
# Hash-based file naming for unique boards
import hashlib
try:
    import numpy as np
except Exception:
    np = None  # Optional dependency; only required for save_training_dataset/load_training_dataset
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from .game import MinesweeperGame, GameState
from .board_metrics import safe_opened_from_visible, status_code, total_safe_cells

# Default keyframe stride is derived from the GUI constants module (scales with board size).
try:
    # NOTE: This is a light dependency (no Qt import), just shared constants/helpers.
    from .gui.constants import keyframe_stride_for_cells, keyframes_for_cells
except Exception:
    keyframe_stride_for_cells = None
    keyframes_for_cells = None


class StateCollector:
    """
    I use this class to collect and store game states on disk.
    """
    
    def __init__(self, output_dir: str = "data/game_states", auto_save: bool = True, keyframe_stride: Optional[int] = None):
        """
        Initialize the state collector.
        
        Args:
            output_dir: Directory to save collected states
            auto_save: If True, save states immediately after each capture
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.states: List[Dict] = []
        self.current_game_id: Optional[int] = None
        self.current_filename: Optional[str] = None
        self.current_filepath: Optional[Path] = None
        self.auto_save = auto_save
        # Option A (recommended): store full visible-board snapshots only periodically.
        # Every action is stored, but only every N actions include a full "board" keyframe.
        self._auto_keyframing = (keyframe_stride is None)
        if keyframe_stride is None:
            # Placeholder; will be derived per game in start_game().
            keyframe_stride = 25
        self.keyframe_stride = max(1, int(keyframe_stride))
        self.game_metadata: Optional[Dict] = None
        # Root-level flags: only flags correctly placed on top of mines.
        # Stored as a list of [row, col] pairs in the saved JSON.
        self.flags_on_mines: List[List[int]] = []
        # Cache the latest visible board snapshot so we can always include a board for the
        # *last* state in on-disk JSON, even when keyframing is enabled and the game is
        # still in progress.
        self._last_visible_board: List[List[str]] = []
    
    def start_game(self, game_id: Optional[int] = None, 
                   mode: str = "manual", height: int = 22, width: int = 22, 
                   num_mines: int = 80, seed: Optional[int] = None,
                   actual_board: Optional[List[List[str]]] = None):
        """
        Start collecting states for a new game.
        
        Args:
            game_id: Optional ID for this game (auto-generated if None)
            mode: Game mode ("manual", "bot", "demo", etc.)
            height: Board height
            width: Board width
            num_mines: Number of mines
            seed: Random seed (if any)
            actual_board: The actual board configuration (with mine locations)
        """
        # NOTE: We do NOT decide the filename here anymore.
        # The filename must be a stable hash of metadata["actual_board_initial"].
        # For most games, actual_board_initial only becomes available after the first click.
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        created_at = now.isoformat()
        self.current_filename = None
        self.current_filepath = None

        # Derive stride from board size if configured for auto keyframing.
        target_keyframes = None
        if self._auto_keyframing and callable(keyframe_stride_for_cells):
            try:
                cells = int(height) * int(width)
                self.keyframe_stride = int(keyframe_stride_for_cells(cells))
                if callable(keyframes_for_cells):
                    target_keyframes = int(keyframes_for_cells(cells))
            except Exception:
                # Keep existing stride on failure
                target_keyframes = None
        
        # Store game metadata with actual_board
        self.game_metadata = {
            # Schema version for game_states JSON
            "schema_version": 2,
            # Keyframing parameters (Option A)
            "keyframe_stride": int(self.keyframe_stride),
            "target_keyframes": int(target_keyframes) if target_keyframes is not None else None,
            # Explicit file-level counters (no fallback/derivation needed)
            "mines_triggered": 0,
            # Flag display policy for Saved Game States board UI:
            # - manual: we don't store flag actions, so show all correct flags at once
            # - bots (logic/nn/etc.): show flags progressively via flag actions
            "show_all_flags": (str(mode) == "manual"),
            # Human-oriented timestamps (preferred for display/sorting)
            'created_at': created_at,
            'last_updated': created_at,
            # Back-compat / filename correlation
            'timestamp': timestamp,
            'mode': mode,
            'height': height,
            'width': width,
            'num_mines': num_mines,
            'seed': seed,
            'game_id': game_id,
            'actual_board_initial': actual_board,  # Store actual board in metadata
            # File-level result (kept here; per-state game_state is no longer stored)
            'game_state': GameState.PROG.value,
        }
        
        self.current_game_id = game_id
        self.states = []
        self.flags_on_mines = []
        # Important: do NOT create a file on start_game().
        # We only want to create a game file once at least one state exists.
        # The first write happens on the first capture_state() when auto_save is enabled.

    def _board_hash(self, actual_board_initial: List[List[str]]) -> str:
        """
        Compute a stable hash for a unique game based ONLY on metadata["actual_board_initial"].
        """
        payload = json.dumps(actual_board_initial, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _ensure_hashed_filepath(self):
        """
        Ensure current_filepath/current_filename follow for ALL modes:
          game_<hash>_<mode>_<timestamp>.json

        This avoids special-casing modes and reduces code redundancy.
        """
        if self.current_filepath is not None and self.current_filename:
            return
        if not isinstance(self.game_metadata, dict):
            return
        actual = self.game_metadata.get("actual_board_initial")
        if not actual:
            return
        if not isinstance(actual, list):
            return
        try:
            h = self._board_hash(actual)
        except Exception:
            return

        mode = str(self.game_metadata.get("mode") or "manual")

        # For collision suffix, prefer created_at timestamp, else "now".
        now = datetime.now()
        ts = self.game_metadata.get("created_at") or now.isoformat()
        try:
            ts_suffix = datetime.fromisoformat(ts).strftime("%Y%m%d_%H%M%S")
        except Exception:
            ts_suffix = now.strftime("%Y%m%d_%H%M%S")

        # Always include timestamp (unique by default).
        base = f"game_{h}_{mode}_{ts_suffix}.json"
        path = self.output_dir / base
        if path.exists():
            # Extremely unlikely: if the same timestamp collides too, bump a counter.
            bump = 1
            while (self.output_dir / f"game_{h}_{mode}_{ts_suffix}_{bump}.json").exists():
                bump += 1
            base = f"game_{h}_{mode}_{ts_suffix}_{bump}.json"
            path = self.output_dir / base
        self.current_filename = base
        self.current_filepath = path
        return

    def export_precomputed_states(
        self,
        *,
        mode: str,
        height: int,
        width: int,
        num_mines: int,
        actual_board_initial: List[List[str]],
        states: List[Dict],
        flags_on_mines: Optional[List[List[int]]] = None,
        final_game_state: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Export a NEW game-states JSON file for a precomputed sequence (e.g., Logic Bot),
        without disrupting any currently active collection session.
        """
        if not states:
            return None

        snapshot = (
            self.states,
            self.game_metadata,
            self.current_filename,
            self.current_filepath,
            self.current_game_id,
            self.flags_on_mines,
        )
        try:
            self.start_game(
                mode=mode,
                height=int(height),
                width=int(width),
                num_mines=int(num_mines),
                seed=None,
                actual_board=actual_board_initial,
            )
            # Compress into keyframes (Option A): keep action for every step, and only keep
            # board snapshots every keyframe_stride steps (always include index 0).
            compressed: List[Dict] = []
            stride = int(self.keyframe_stride) if int(self.keyframe_stride) > 0 else 1
            last_mines_triggered = 0
            total_n = len(states)
            last_board_snapshot = None
            for i, s in enumerate(states):
                if not isinstance(s, dict):
                    continue
                # Keep per-action mines_triggered so keyframing does not lose this signal.
                mt = s.get("mines_triggered")
                if mt is not None:
                    try:
                        last_mines_triggered = int(mt)
                    except Exception:
                        pass
                rec: Dict = {
                    "action": s.get("action"),
                    "mines_triggered": int(last_mines_triggered),
                }
                # Always include the first board snapshot if present.
                # Also always include the FINAL board snapshot so the last action is inspectable
                # even when keyframing is enabled.
                if i == 0 or (i % stride == 0) or (i == total_n - 1):
                    if s.get("board") is not None:
                        rec["board"] = s.get("board")
                if i == total_n - 1:
                    last_board_snapshot = s.get("board")
                compressed.append(rec)
            self.states = compressed
            self.flags_on_mines = flags_on_mines or []
            if isinstance(self.game_metadata, dict):
                self.game_metadata["mines_triggered"] = int(last_mines_triggered)
                # Logic-bot and other precomputed/bot exports should always continue after mine triggers.
                # Make this explicit in metadata so UI status logic is correct.
                try:
                    self.game_metadata["allow_mine_triggers"] = (str(mode) != "manual")
                except Exception:
                    self.game_metadata["allow_mine_triggers"] = True
            # Ensure root-level last_board is populated for precomputed exports too.
            try:
                self._last_visible_board = [row[:] for row in (last_board_snapshot or [])]
            except Exception:
                self._last_visible_board = []
            # Compute and store UI-friendly file-level status code.
            if isinstance(self.game_metadata, dict):
                try:
                    self.game_metadata["game_state"] = self._compute_status_code(
                        allow_mine_triggers=bool(self.game_metadata.get("allow_mine_triggers", False)),
                        mines_triggered=int(self.game_metadata.get("mines_triggered", 0) or 0),
                        visible_board=self._last_visible_board,
                        height=int(self.game_metadata.get("height", height) or height),
                        width=int(self.game_metadata.get("width", width) or width),
                        num_mines=int(self.game_metadata.get("num_mines", num_mines) or num_mines),
                    )
                except Exception:
                    self.game_metadata["game_state"] = "PROG"
            # Ensure hashed path is selected immediately (actual_board_initial is known).
            self._ensure_hashed_filepath()
            if self.auto_save:
                self._save_to_file()
            else:
                self._save_to_file()
            return self.current_filepath
        finally:
            (
                self.states,
                self.game_metadata,
                self.current_filename,
                self.current_filepath,
                self.current_game_id,
                self.flags_on_mines,
            ) = snapshot

    def resume_game_file(self, filepath: str) -> bool:
        """
        Resume collecting states into an existing game JSON file.
        This loads metadata + existing states and ensures future captures append
        to the same file (instead of creating a new one).
        """
        try:
            path = Path(filepath)
            with open(path, "r") as f:
                data = json.load(f)

            states = data.get("states", [])
            metadata = data.get("metadata", None)

            if not isinstance(states, list) or metadata is None:
                return False

            self.current_filepath = path
            self.current_filename = path.name
            self.game_metadata = metadata
            # When resuming, respect the file's configured keyframing stride.
            try:
                if isinstance(self.game_metadata, dict) and self.game_metadata.get("keyframe_stride") is not None:
                    self.keyframe_stride = max(1, int(self.game_metadata.get("keyframe_stride")))
                    self._auto_keyframing = False
            except Exception:
                pass
            # Ensure we have plain dicts in-memory.
            # Also migrate old schema in-place by stripping redundant keys we no longer write.
            cleaned = []
            for s in states:
                if not isinstance(s, dict):
                    continue
                s.pop("statistics", None)
                s.pop("game_state", None)
                s.pop("diff", None)
                if isinstance(s.get("state"), dict):
                    s["state"].pop("statistics", None)
                    s["state"].pop("game_state", None)
                    s["state"].pop("diff", None)
                cleaned.append(s)
            self.states = cleaned

            # Load existing root-level flags (if present)
            flags = data.get("flags_on_mines", [])
            if isinstance(flags, list):
                # Normalize to list[list[int]]
                out: List[List[int]] = []
                for item in flags:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        try:
                            out.append([int(item[0]), int(item[1])])
                        except Exception:
                            pass
                self.flags_on_mines = out
            else:
                self.flags_on_mines = []

            # Ensure metadata has created_at / last_updated / game_state
            now_iso = datetime.now().isoformat()
            if isinstance(self.game_metadata, dict):
                # Back-fill keyframing metadata for older files
                if "schema_version" not in self.game_metadata:
                    self.game_metadata["schema_version"] = 1
                if "keyframe_stride" not in self.game_metadata:
                    self.game_metadata["keyframe_stride"] = int(self.keyframe_stride)
                if "mines_triggered" not in self.game_metadata:
                    # Best-effort default; older files won't have this.
                    self.game_metadata["mines_triggered"] = 0
                if "show_all_flags" not in self.game_metadata:
                    # Default policy: manual shows all correct flags; other modes show progressive flags.
                    mode = str(self.game_metadata.get("mode") or "")
                    self.game_metadata["show_all_flags"] = (mode == "manual")
                if not self.game_metadata.get("created_at"):
                    ts = self.game_metadata.get("timestamp")
                    created = None
                    if isinstance(ts, str):
                        try:
                            created = datetime.strptime(ts, "%Y%m%d_%H%M%S").isoformat()
                        except Exception:
                            created = None
                    self.game_metadata["created_at"] = created or now_iso

                # Prefer root last_updated if present
                self.game_metadata["last_updated"] = (
                    self.game_metadata.get("last_updated")
                    or data.get("last_updated")
                    or now_iso
                )

            # File-level status (UI-friendly). Will be updated on every capture.
            self.game_metadata["game_state"] = self.game_metadata.get("game_state") or "PROG"

            return True
        except Exception:
            return False

    @staticmethod
    def _compute_status_code(
        *,
        allow_mine_triggers: bool,
        mines_triggered: int,
        visible_board: List[List[str]],
        height: int,
        width: int,
        num_mines: int,
    ) -> str:
        """
        Compute the file-level status code using the project's single source of truth
        (`minesweeper.board_metrics.status_code`).

        Status codes:
        - WON:  100% progress and mines_triggered == 0
        - DONE: 100% progress and mines_triggered > 0
        - LOST: mines_triggered > 0 and progress < 100%
        - PROG: otherwise
        """
        ts = total_safe_cells(height=int(height), width=int(width), num_mines=int(num_mines))
        so = safe_opened_from_visible(visible_board)
        mt = int(mines_triggered) if mines_triggered is not None else 0
        return status_code(mines_triggered=int(mt), safe_opened=int(so), total_safe=int(ts))

    def update_flags_on_mines(self, game: MinesweeperGame, flagged_grid: List[List[bool]]):
        """
        Update the root-level list of *correct* flags (flags placed on mines).
        This is meant to be called by the GUI after a right-click (and after the first click
        when the actual board becomes available).
        """
        try:
            actual = game.get_actual_board()
            if not actual:
                return

            h = len(actual)
            w = len(actual[0]) if actual and actual[0] else 0

            out: List[List[int]] = []
            for r in range(h):
                fg_row = flagged_grid[r] if r < len(flagged_grid) else []
                act_row = actual[r] if r < len(actual) else []
                for c in range(w):
                    is_flagged = bool(fg_row[c]) if c < len(fg_row) else False
                    is_mine = (act_row[c] == "M") if c < len(act_row) else False
                    if is_flagged and is_mine:
                        out.append([r, c])

            self.flags_on_mines = out

            # Persist immediately if auto-save is enabled and a file exists.
            if self.auto_save:
                self._save_to_file()
        except Exception:
            # Never break gameplay due to collection
            return
    
    def end_game(self):
        """End the current game collection."""
        # This method exists for compatibility
        pass
    
    def capture_state(self, game: MinesweeperGame, action: Optional[tuple] = None):
        """
        Capture the current state of a game.
        
        Args:
            game: The MinesweeperGame instance
            action: Optional tuple (row, col) of the action that led to this state
        """
        # Get visible board
        # IMPORTANT: MinesweeperGame.get_visible_board() returns the live internal board.
        # We must deep-copy it here; otherwise all prior snapshots will mutate into the final state.
        visible_board_live = game.get_visible_board()
        visible_board = [row[:] for row in visible_board_live] if visible_board_live else []
        # Keep a cached copy for write-time "last state always has board" guarantee.
        self._last_visible_board = [row[:] for row in visible_board] if visible_board else []
        
        # Get actual board for ground truth (only store once in metadata)
        actual_board = game.get_actual_board()
        
        # Store actual_board in metadata on first capture if not already set
        if self.game_metadata and not self.game_metadata.get('actual_board_initial') and actual_board:
            self.game_metadata['actual_board_initial'] = [row[:] for row in actual_board]
        # Once we have actual_board_initial, we can lock in the hashed filename.
        self._ensure_hashed_filepath()

        # Update file-level metadata + last_updated
        if isinstance(self.game_metadata, dict):
            try:
                self.game_metadata["mines_triggered"] = int(getattr(game, "mines_triggered", 0) or 0)
            except Exception:
                self.game_metadata["mines_triggered"] = 0
            if not self.game_metadata.get("created_at"):
                self.game_metadata["created_at"] = datetime.now().isoformat()
            self.game_metadata["last_updated"] = datetime.now().isoformat()
        
        # Record allow_mine_triggers explicitly in metadata (always present on disk).
        # This must reflect the current game configuration, so we update it every capture.
        if isinstance(self.game_metadata, dict):
            try:
                self.game_metadata["allow_mine_triggers"] = bool(getattr(game, "allow_mine_triggers", False))
            except Exception:
                self.game_metadata["allow_mine_triggers"] = False

            # Compute and store UI-friendly file-level status code.
            try:
                self.game_metadata["game_state"] = self._compute_status_code(
                    allow_mine_triggers=bool(self.game_metadata.get("allow_mine_triggers", False)),
                    mines_triggered=int(self.game_metadata.get("mines_triggered", 0) or 0),
                    visible_board=visible_board,
                    height=int(self.game_metadata.get("height", len(visible_board)) or len(visible_board)),
                    width=int(self.game_metadata.get("width", len(visible_board[0]) if visible_board else 0) or 0),
                    num_mines=int(self.game_metadata.get("num_mines", 0) or 0),
                )
            except Exception:
                self.game_metadata["game_state"] = "PROG"

        # Option A: action log + periodic keyframes
        # - Always store "action"
        # - Only store full "board" snapshot every N steps (and always at step 0)
        idx = len(self.states)
        mt = 0
        try:
            mt = int(getattr(game, "mines_triggered", 0) or 0)
        except Exception:
            mt = 0
        state_record: Dict = {
            "action": list(action) if isinstance(action, tuple) else action,
            "mines_triggered": mt,
        }
        # Keyframing (Option A): store board periodically, but ALWAYS store a board on terminal
        # states so the last action is directly inspectable in JSON.
        is_terminal = False
        try:
            is_terminal = (game.get_game_state() != GameState.PROG)
        except Exception:
            is_terminal = False
        if idx == 0 or (idx % int(self.keyframe_stride) == 0) or is_terminal:
            state_record["board"] = visible_board
        
        self.states.append(state_record)
        
        # Auto-save immediately after capturing state
        if self.auto_save:
            self._save_to_file()
    
    def _char_to_num(self, char: str) -> int:
        """Convert board character to number for training."""
        if char == "E":
            return -2  # Unrevealed
        elif char == "M":
            return -1  # Mine
        elif char == "B":
            return 0   # Blank (empty revealed)
        else:
            try:
                return int(char)  # Clue number
            except:
                return -2
    
    def _save_to_file(self):
        """Save current states to file (internal method for auto-save)."""
        if not self.current_filename and not self.current_filepath:
            # If we can determine the hashed filename now, do it.
            self._ensure_hashed_filepath()
        if not self.current_filename and not self.current_filepath:
            return

        filepath = self.current_filepath or (self.output_dir / self.current_filename)

        # Keep metadata timestamps consistent on write
        now_iso = datetime.now().isoformat()
        if isinstance(self.game_metadata, dict):
            if not self.game_metadata.get("created_at"):
                self.game_metadata["created_at"] = now_iso
            self.game_metadata["last_updated"] = now_iso
            # Ensure allow_mine_triggers is always present in metadata on disk.
            if "allow_mine_triggers" not in self.game_metadata:
                self.game_metadata["allow_mine_triggers"] = False
        
        # Serialize states. With keyframing, many steps omit "board" (intentionally).
        # For UI/inspection, we additionally store a root-level snapshot `last_board`
        # so the latest action is inspectable without reconstructing.
        states_json = self._convert_to_json_serializable(self.states)

        # Create complete game data with metadata
        game_data = {
            'metadata': self.game_metadata,
            'states': states_json,
            # Root-level snapshot of the most recent visible board (always present when possible).
            'last_board': self._convert_to_json_serializable(self._last_visible_board),
            'flags_on_mines': self._convert_to_json_serializable(self.flags_on_mines),
            'total_states': len(self.states),
            'last_updated': now_iso
        }
        
        with open(filepath, 'w') as f:
            json.dump(game_data, f, indent=2)
    
    def save_game(self, filename: Optional[str] = None):
        """
        Save all collected states for the current game.
        If auto_save is enabled, this just ensures the file is up to date.
        
        Args:
            filename: Optional custom filename (if None, uses auto-generated name)
        """
        if not self.states:
            return
        
        if filename:
            # Use custom filename
            filepath = self.output_dir / filename
            states_json = self._convert_to_json_serializable(self.states)
            game_data = {
                'metadata': self.game_metadata,
                'states': states_json,
                'last_board': self._convert_to_json_serializable(self._last_visible_board),
                'flags_on_mines': self._convert_to_json_serializable(self.flags_on_mines),
                'total_states': len(self.states),
                'last_updated': datetime.now().isoformat()
            }
            with open(filepath, 'w') as f:
                json.dump(game_data, f, indent=2)
            print(f"Saved {len(self.states)} states to {filepath}")
        elif self.auto_save:
            # File already saved, just confirm
            if self.current_filename:
                print(f"Game states auto-saved: {self.current_filename} ({len(self.states)} states)")
            else:
                self._save_to_file()
        else:
            # Manual save without auto-save
            if not self.current_filename:
                # Generate filename if not set
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.current_filename = f"game_{timestamp}.json"
            self._save_to_file()
            print(f"Saved {len(self.states)} states to {self.current_filename}")
    
    def _convert_to_json_serializable(self, data):
        """Convert data to JSON-serializable formats."""
        if isinstance(data, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_json_serializable(item) for item in data]
        else:
            return data
    
    def load_game(self, filename: str) -> Dict:
        """
        Load a saved game's states.
        
        Returns:
            Dictionary with 'metadata', 'states', 'total_states', 'last_updated'
        """
        filepath = self.output_dir / filename
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_training_dataset(self, filename: str = "training_dataset.npz"):
        """Save collected states as a numpy dataset for training."""
        if np is None:
            raise ModuleNotFoundError("numpy is required for save_training_dataset(). Install numpy or use the JSON/NPZ exporter script.")
        if not self.states:
            raise ValueError("No states collected. Capture some states first.")
        
        inputs = []
        target_mines = []
        target_safe = []
        
        for state_record in self.states:
            training = state_record['training']
            inputs.append(np.array(training['input']))
            target_mines.append(np.array(training['target_mines']))
            target_safe.append(np.array(training['target_safe']))
        
        inputs_array = np.stack(inputs)
        target_mines_array = np.stack(target_mines)
        target_safe_array = np.stack(target_safe)
        
        filepath = self.output_dir / filename
        np.savez_compressed(
            filepath,
            inputs=inputs_array,
            target_mines=target_mines_array,
            target_safe=target_safe_array
        )
        
        print(f"Saved training dataset with {len(self.states)} samples to {filepath}")
    
    def load_training_dataset(self, filename: str = "training_dataset.npz") -> Dict:
        """Load a saved training dataset."""
        if np is None:
            raise ModuleNotFoundError("numpy is required for load_training_dataset(). Install numpy or use the JSON/NPZ exporter script.")
        filepath = self.output_dir / filename
        data = np.load(filepath)
        return {
            'inputs': data['inputs'],
            'target_mines': data['target_mines'],
            'target_safe': data['target_safe']
        }
