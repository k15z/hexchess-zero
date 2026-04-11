"""Type stubs for the `hexchess` Rust PyO3 bindings.

Hand-maintained. The surface covered here is the subset the Python training
code (`training/`) uses — not every attribute the Rust module exposes. If you
add a new binding and the training code imports it, add it here so `ty` can
type-check usages.

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

    @staticmethod
    def from_notation(notation: str) -> "Move": ...


class RankedMove:
    move: Move
    score: int


class Piece:
    color: str
    piece: str
    q: int
    r: int
    square: int


class Game:
    def __init__(self) -> None: ...
    def apply(self, move: Move) -> None: ...
    def apply_move(self, move: Move) -> None: ...
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
        model_path: str | None = ...,
        eval_mode: bool = ...,
        **kwargs: Any,
    ) -> None: ...
    def run(self, game: Game, temperature: float = ...) -> MctsResult: ...
    def run_pcr(self, game: Game, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    def aux_opponent_policy(self) -> np.ndarray: ...
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
def from_notation(notation: str) -> Move: ...
def to_notation(move: Move) -> str: ...
def index_to_move(index: int) -> Move: ...
def move_to_index(
    from_q: int,
    from_r: int,
    to_q: int,
    to_r: int,
    promotion: str | None = ...,
) -> int: ...
def mirror_indices_array() -> list[int]: ...
def num_move_indices() -> int: ...
def minimax_search(
    game: Game,
    depth: int,
    weights: EvalWeights | None = ...,
    **kwargs: Any,
) -> MinimaxResult: ...
def minimax_search_all(
    game: Game,
    depth: int,
    weights: EvalWeights | None = ...,
    **kwargs: Any,
) -> MinimaxAllResult: ...
def minimax_search_with_policy(
    game: Game,
    depth: int,
    weights: EvalWeights | None = ...,
    **kwargs: Any,
) -> MinimaxPolicyResult: ...
