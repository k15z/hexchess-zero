"""Type stubs for the `hexchess` Rust PyO3 bindings.

Hand-maintained. If you add a Rust binding and the training code starts
referencing it, update this file so `ty` can type-check usages — the drift
guard at `training/test_hexchess_stub.py` enforces *presence* (name exists)
but not signature accuracy, so wrong signatures here will silently mask
real bugs. When in doubt, check `bindings/python/src/lib.rs` for the
`#[pyo3(signature = ...)]` attribute.

The native module lives in `bindings/python/src/lib.rs`; `maturin develop`
installs it into the project venv as the top-level `hexchess` package.
"""

from typing import Any

import numpy as np

NUM_MOVE_INDICES: int
TENSOR_SHAPE: tuple[int, int, int]


class Move:
    from_q: int
    from_r: int
    to_q: int
    to_r: int
    promotion: str | None
    notation: str
    # Tuple aliases exposed on the Rust type.
    from_: tuple[int, int]
    to: tuple[int, int]

    @staticmethod
    def from_notation(notation: str) -> "Move":
        """Parse a move from Glinski notation like 'f5-f6' or 'f10-f11=Q'."""
        ...


class RankedMove:
    move: Move
    score: int


class Piece:
    color: str
    piece: str
    q: int
    r: int
    # `square` is the Glinski notation for the cell (e.g. "b1"), not a numeric index.
    square: str


class Game:
    def __init__(self) -> None: ...
    def apply(self, move: Move | str) -> None:
        """Apply a move given as a Move object or a Glinski notation string
        (e.g. 'f5-f6', 'f10-f11=Q'). Raises ValueError if the move is illegal."""
        ...
    def apply_move(
        self,
        from_q: int,
        from_r: int,
        to_q: int,
        to_r: int,
        promotion: str | None = ...,
    ) -> None:
        """Legacy five-positional-args API. Prefer `apply(mv)` for new code."""
        ...
    def undo_move(self) -> None: ...
    def clone(self) -> "Game": ...
    def is_game_over(self) -> bool: ...
    def is_in_check(self) -> bool: ...
    def legal_moves(self) -> list[Move]: ...
    def move_count(self) -> int: ...
    def side_to_move(self) -> str: ...
    def status(self) -> str: ...
    def board_state(self) -> list[Piece]: ...


class TtStats:
    clears: int
    current_size: int
    hits: int
    misses: int


class MctsResult:
    best_move: Move
    nodes: int
    policy: np.ndarray
    value: float


class MctsSearch:
    simulations: int

    def __init__(
        self,
        simulations: int = ...,
        c_puct: float | None = ...,
        model_path: str | None = ...,
        batch_size: int = ...,
        tt_capacity: int = ...,
        intra_threads: int = ...,
        use_weighted_eval: bool = ...,
        dirichlet_epsilon: float = ...,
        dirichlet_alpha: float = ...,
        eval_mode: bool = ...,
    ) -> None: ...
    def run(self, game: Game, temperature: float = ...) -> MctsResult: ...
    def run_pcr(self, game: Game, ply: int = ...) -> dict[str, Any]:
        """Returns {best_move, value, nodes, was_full_search, policy_target (or None), ...}."""
        ...
    def aux_opponent_policy(self) -> np.ndarray | None:
        """Opponent-reply visit distribution from the previous search, or
        `None` if unavailable (no search yet, or best child unexpanded)."""
        ...
    def config_summary(self) -> dict[str, Any]: ...
    def set_resign_enabled(self, enabled: bool) -> None: ...
    def set_rng_seed(self, seed: int) -> None: ...
    def tt_stats(self) -> TtStats: ...


class EvalWeights:
    @staticmethod
    def material_only() -> "EvalWeights": ...


class MinimaxResult:
    best_move: Move
    score: int
    nodes: int


class MinimaxPolicyResult:
    best_move: Move
    best_score: int
    moves: list[RankedMove]
    nodes: int


class MinimaxAllResult:
    moves: list[RankedMove]
    nodes: int


def encode_board(game: Game) -> np.ndarray: ...
def encode_batch(games: list[Game]) -> np.ndarray: ...
def to_notation(q: int, r: int) -> str:
    """Convert axial (q, r) to Glinski notation like 'f6'."""
    ...
def from_notation(s: str) -> tuple[int, int]:
    """Parse Glinski notation like 'f6' into (q, r). Raises ValueError on bad input.

    See `Move.from_notation` for parsing a full move string like 'f5-f6'.
    """
    ...
def index_to_move(index: int) -> Move: ...
def move_to_index(
    from_q: int,
    from_r: int,
    to_q: int,
    to_r: int,
    promotion: str | None = ...,
) -> int: ...
def mirror_indices_array() -> np.ndarray: ...
def num_move_indices() -> int: ...
def minimax_search(
    game: Game,
    depth: int,
    weights: EvalWeights | None = ...,
) -> MinimaxResult: ...
def minimax_search_all(
    game: Game,
    depth: int,
    weights: EvalWeights | None = ...,
) -> MinimaxAllResult: ...
def minimax_search_with_policy(
    game: Game,
    depth: int,
    weights: EvalWeights | None = ...,
) -> MinimaxPolicyResult: ...
