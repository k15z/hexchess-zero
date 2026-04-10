"""v2 .npz loader and target builder for the chunk 6 trainer.

The worker (chunk 5) writes rich per-position arrays:

    boards                (N, 22, 11, 11)        int8
    policy                (N, num_move_indices)  float16
    policy_aux_opp        (N, num_move_indices)  float16
    wdl_terminal          (N, 3)                 float32
    wdl_short             (N, 3)                 float32
    mlh                   (N,)                   int16
    was_full_search       (N,)                   bool
    root_q, root_n, root_entropy, nn_value_at_position, legal_count, ply, game_id

This module provides pure functions to:
  - load one v2 .npz into numpy arrays (filtered to was_full_search=True),
  - build the per-batch ``targets`` dict expected by ``compute_losses``,
  - derive a legal_mask from the policy visits.

The loader is deliberately numpy-only; the trainer wraps a batch in
torch tensors at dataloader collate time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# Mirror move-index table from the Rust engine. The mirror used is central
# inversion (q,r) → (-q,-r) — the only hex involution that preserves Glinski's
# pawn directions and promotion edges. See engine/src/serialization.rs.
try:
    import hexchess as _hexchess

    MIRROR_INDICES: np.ndarray | None = np.asarray(
        _hexchess.mirror_indices_array(), dtype=np.int64
    )
except (ImportError, AttributeError):  # binding not built or pre-mirror
    MIRROR_INDICES = None


@dataclass
class V2Batch:
    """A slice of v2 training samples as plain numpy arrays (fp32 / bool)."""

    boards: np.ndarray          # (B, 22, 11, 11) float32
    policy: np.ndarray          # (B, num_moves)  float32
    aux_policy: np.ndarray      # (B, num_moves)  float32
    wdl_terminal: np.ndarray    # (B, 3)          float32
    wdl_short: np.ndarray       # (B, 3)          float32
    mlh: np.ndarray             # (B,)            float32
    legal_mask: np.ndarray      # (B, num_moves)  bool
    ply: np.ndarray             # (B,)            int32

    def __len__(self) -> int:
        return int(self.boards.shape[0])


def _decode_boards(boards_int8: np.ndarray) -> np.ndarray:
    """Cast int8 board tensor back to float32.

    The worker stored ``rint(float_board).astype(int8)``. The 12 piece
    planes and the binary meta planes (side-to-move, en-passant, in-check,
    last-move-from/to, repetition, validity) round-trip exactly since
    every value is in {0, 1, 2}.

    Four meta planes (fullmove/100, halfmove/100, repetition, halfmove/50)
    were in the fractional range [0, ~2] and were lossily rounded to
    integers. We accept that loss here — see the worker module docstring
    for the rationale. No explicit rescale is applied (the float values
    post-round-trip are the integer-rounded versions).
    """
    return boards_int8.astype(np.float32)


def _legal_mask_from_policy(policy: np.ndarray) -> np.ndarray:
    """Derive a legal_mask from the MCTS visit distribution.

    Nonzero entries in the target policy correspond to visited moves, and
    only legal moves are ever visited. Illegal entries are 0 in the
    target so we can safely mask them out in the predicted distribution.
    This is a conservative approximation documented in notes/13 §4:
    legal-but-unvisited moves are marked "illegal" for the loss, which is
    fine because the soft-target CE already has zero weight on them.
    """
    return policy > 0.0


def load_imitation_npz(path: str | Path) -> V2Batch:
    """Load a legacy imitation .npz (boards/policies/outcomes) as a V2Batch.

    Missing fields are filled with sensible defaults:
    - aux_policy = uniform over legal moves (no opponent data in imitation)
    - wdl_short = wdl_terminal (terminal outcome is the best horizon-8 proxy)
    - mlh = 0 (no game-length info in legacy format)

    This lets the trainer mix imitation data into the self-play replay buffer
    to anchor the policy to the minimax teacher signal during early training.
    Without this, weak self-play can dilute the bootstrap and the model
    regresses (observed: v2 was WORSE than heuristic despite v1 being strong).
    """
    path = Path(path)
    data = np.load(str(path))

    boards = np.asarray(data["boards"]).astype(np.float32)
    policy = np.asarray(data["policies"]).astype(np.float32)
    wdl = np.asarray(data["outcomes"]).astype(np.float32)
    n = boards.shape[0]

    # Uniform aux over legal moves (policy > 0 marks visited/legal moves)
    legal_mask = policy > 0.0
    legal_counts = legal_mask.sum(axis=1, keepdims=True).clip(min=1)
    aux_policy = (legal_mask.astype(np.float32) / legal_counts)

    return V2Batch(
        boards=boards,
        policy=policy,
        aux_policy=aux_policy,
        wdl_terminal=wdl,
        wdl_short=wdl,  # best proxy: terminal outcome
        mlh=np.zeros(n, dtype=np.float32),
        legal_mask=legal_mask,
        ply=np.zeros(n, dtype=np.int32),
    )


def load_v2_npz(path: str | Path) -> V2Batch:
    """Load a chunk-5 v2 .npz file and return an fp32 V2Batch.

    Only rows with ``was_full_search == True`` are kept. The loader uses
    ``np.load`` (not mmap) because we copy every array to fp32 anyway.
    """
    path = Path(path)
    data = np.load(str(path))

    was_full = np.asarray(data["was_full_search"]).astype(bool)
    idx = np.nonzero(was_full)[0]
    if idx.size == 0:
        raise ValueError(f"no full-search rows in {path}")

    boards = _decode_boards(np.asarray(data["boards"])[idx])
    policy = np.asarray(data["policy"])[idx].astype(np.float32)
    aux_policy = np.asarray(data["policy_aux_opp"])[idx].astype(np.float32)
    wdl_terminal = np.asarray(data["wdl_terminal"])[idx].astype(np.float32)
    wdl_short = np.asarray(data["wdl_short"])[idx].astype(np.float32)
    mlh = np.asarray(data["mlh"])[idx].astype(np.float32)
    ply = np.asarray(data["ply"])[idx].astype(np.int32)
    legal_mask = _legal_mask_from_policy(policy)

    return V2Batch(
        boards=boards,
        policy=policy,
        aux_policy=aux_policy,
        wdl_terminal=wdl_terminal,
        wdl_short=wdl_short,
        mlh=mlh,
        legal_mask=legal_mask,
        ply=ply,
    )


def mirror_batch(batch: V2Batch, *, apply_to_boards: bool = True) -> V2Batch:
    """Horizontally mirror a v2 batch via central hex inversion.

    Mirrors policy and aux_policy via the precomputed Rust ``MIRROR_INDICES``
    table. Mirrors boards by flipping both row and column axes of the
    11x11 grid embedding (central inversion `(q,r)→(-q,-r)` corresponds to
    flipping both axes since the encoder uses `col=q+5, row=r+5`).

    The board mirror is enabled by default — central inversion preserves
    `is_valid()` for all 91 cells trivially, and the encoder pre-flips by
    side-to-move so piece-plane indices need no swap. WDL and MLH targets
    are unchanged.

    Raises if `MIRROR_INDICES` is unavailable (binding not rebuilt).
    """
    if MIRROR_INDICES is None:
        raise RuntimeError(
            "mirror_indices_array() not available — rebuild the Python "
            "binding with `make setup` to enable mirror augmentation."
        )

    new_policy = batch.policy[:, MIRROR_INDICES]
    new_aux = batch.aux_policy[:, MIRROR_INDICES]
    new_legal = batch.legal_mask[:, MIRROR_INDICES]
    new_boards = batch.boards
    if apply_to_boards:
        new_boards = batch.boards[:, :, ::-1, ::-1].copy()
    return replace(
        batch,
        boards=new_boards,
        policy=new_policy,
        aux_policy=new_aux,
        legal_mask=new_legal,
    )


def maybe_mirror_batch(batch: V2Batch, p: float, rng: np.random.Generator) -> V2Batch:
    """With probability `p`, return `mirror_batch(batch)`; else `batch`."""
    if MIRROR_INDICES is None or p <= 0.0:
        return batch
    if rng.random() < p:
        return mirror_batch(batch)
    return batch


def build_targets_dict(batch: V2Batch) -> dict:
    """Build the ``targets`` dict argument for :func:`training.losses.compute_losses`.

    The returned dict contains only the target arrays (still numpy); the
    trainer is responsible for converting to torch tensors and moving to
    the target device.
    """
    return {
        "policy": batch.policy,
        "wdl": batch.wdl_terminal,
        "mlh": batch.mlh,
        "stv": batch.wdl_short,
        "aux_policy": batch.aux_policy,
    }
