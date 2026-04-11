"""Type stubs for the Rust-backed ``hexchess`` Python bindings.

Mirrors the public API exported by ``bindings/python/src/lib.rs``. Keep in
sync when adding or renaming pyclass/pyfunction members there.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class Move:
    from_q: int
    from_r: int
    to_q: int
    to_r: int
    promotion: str | None
    notation: str
    from_: tuple[int, int]
    to: tuple[int, int]

    def __init__(
        self,
        from_q: int,
        from_r: int,
        to_q: int,
        to_r: int,
        promotion: str | None = ...,
    ) -> None: ...
    @staticmethod
    def from_notation(s: str) -> "Move": ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


class Piece:
    q: int
    r: int
    piece: str
    color: str
    square: str | None


class TtStats:
    hits: int
    misses: int
    clears: int
    current_size: int


class RankedMove:
    move: Move
    score: int


class MctsResult:
    best_move: Move
    value: float
    nodes: int

    @property
    def policy(self) -> NDArray[np.float32]: ...


class MinimaxResult:
    best_move: Move
    score: int
    nodes: int


class MinimaxAllResult:
    moves: list[RankedMove]
    nodes: int


class MinimaxPolicyResult:
    best_move: Move
    best_score: int
    moves: list[RankedMove]
    nodes: int


class EvalWeights:
    def __init__(
        self,
        material: int = ...,
        mobility: int = ...,
        pawn_advance: int = ...,
        center_control: int = ...,
        king_safety: int = ...,
        bishop_color_bonus: int = ...,
        pawn_connected: int = ...,
        pawn_isolated: int = ...,
        passed_pawn: int = ...,
        king_tropism: int = ...,
    ) -> None: ...
    @staticmethod
    def material_only() -> "EvalWeights": ...


class Game:
    def __init__(self) -> None: ...
    def legal_moves(self) -> list[Move]: ...
    def apply_move(
        self,
        from_q: int,
        from_r: int,
        to_q: int,
        to_r: int,
        promotion: str | None = ...,
    ) -> None: ...
    def apply(self, mv: Move | str) -> None: ...
    def undo_move(self) -> None: ...
    def status(self) -> str: ...
    def is_game_over(self) -> bool: ...
    def side_to_move(self) -> str: ...
    def move_count(self) -> int: ...
    def board_state(self) -> list[Piece]: ...
    def is_in_check(self) -> bool: ...
    def clone(self) -> "Game": ...
    def __copy__(self) -> "Game": ...
    def __deepcopy__(self, memo: Any) -> "Game": ...
    def __str__(self) -> str: ...


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
    def set_rng_seed(self, seed: int) -> None: ...
    def set_resign_enabled(self, enabled: bool) -> None: ...
    def run_pcr(self, game: Game, ply: int = ...) -> dict[str, Any]: ...
    def run(self, game: Game, temperature: float = ...) -> MctsResult: ...
    def aux_opponent_policy(self) -> NDArray[np.float32] | None: ...
    def tt_stats(self) -> TtStats: ...
    def config_summary(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


def encode_board(game: Game) -> NDArray[np.float32]: ...
def encode_batch(games: list[Game]) -> NDArray[np.float32]: ...
def move_to_index(
    from_q: int,
    from_r: int,
    to_q: int,
    to_r: int,
    promotion: str | None = ...,
) -> int: ...
def index_to_move(idx: int) -> Move: ...
def num_move_indices() -> int: ...
def mirror_indices_array() -> NDArray[np.uint32]: ...
def minimax_search(
    game: Game, depth: int, weights: EvalWeights | None = ...,
) -> MinimaxResult: ...
def minimax_search_all(
    game: Game, depth: int, weights: EvalWeights | None = ...,
) -> MinimaxAllResult: ...
def minimax_search_with_policy(
    game: Game, depth: int, weights: EvalWeights | None = ...,
) -> MinimaxPolicyResult: ...
def to_notation(q: int, r: int) -> str: ...
def from_notation(s: str) -> tuple[int, int]: ...


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


TENSOR_SHAPE: tuple[int, int, int]
NUM_MOVE_INDICES: int
