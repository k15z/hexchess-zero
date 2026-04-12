"""v2 .npz loader and target builder for the chunk 6 trainer.

The worker (chunk 5) writes rich per-position arrays:

    boards                (N, 22, 11, 11)        float16
    policy                (N, num_move_indices)  float16
    policy_aux_opp        (N, num_move_indices)  float16
    legal_mask            (N, num_move_indices)  bool
    wdl_terminal          (N, 3)                 float32
    wdl_short             (N, 3)                 float32
    mlh                   (N,)                   int16
    was_full_search       (N,)                   bool
    root_q, root_n, root_entropy, nn_value_at_position, legal_count, ply, game_id

This module provides pure functions to:
  - load one v2 .npz into numpy arrays (filtered to was_full_search=True),
  - build the per-batch ``targets`` dict expected by ``compute_losses``.

``legal_mask`` is read directly from the .npz and represents **legality**,
not visit counts. This matters: if the mask were derived as ``policy > 0``
(visit mask), legal-but-unvisited moves would be dropped from the policy
softmax denominator and the network would get no loss signal pushing it
to *not* place mass on them — a silent policy-learning degradation at low
sim counts. The worker writes the true legality bitmap so this module
never has to guess.

The loader is deliberately numpy-only; the trainer wraps a batch in
torch tensors at dataloader collate time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# NOTE: mirror augmentation was removed when the encoder switched to the
# side-to-move frame (engine::serialization::encode_board). In STM frame,
# the "mirrored" version of any position is encoded identically to the
# original — the color symmetry is baked in, so the 2x data multiplier is
# no longer needed (and the previous absolute-frame implementation was
# silently corrupting pawn/promotion targets — see the original
# mirror_batch bug, commit history).


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


def _decode_boards(boards: np.ndarray) -> np.ndarray:
    """Cast persisted board tensor to float32 for training.

    Current schema stores boards as float16 to preserve all fractional input
    planes while reducing storage size. This helper simply upcasts to float32
    for stable math in the model.
    """
    return boards.astype(np.float32)


_IMITATION_REQUIRED_KEYS = ("boards", "policies", "legal_masks", "outcomes")
_V2_REQUIRED_KEYS = (
    "boards",
    "policy",
    "policy_aux_opp",
    "legal_mask",
    "wdl_terminal",
    "wdl_short",
    "mlh",
    "was_full_search",
    "ply",
)


def _check_required_keys(
    data: np.lib.npyio.NpzFile, required: tuple[str, ...], path: Path
) -> None:
    """Raise a clear schema error if any required key is missing.

    Stale pre-legal_mask data would otherwise surface as a bare ``KeyError``
    that trainer iterators happily swallow. Giving the error a structured
    message and a specific list of missing keys makes the failure mode
    obvious when a regenerate is needed.
    """
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(
            f"{path} is missing required field(s) {missing}. This .npz was "
            "written with an older schema — regenerate it after the "
            "legal_mask change."
        )


def load_imitation_npz(path: str | Path) -> V2Batch:
    """Load an imitation .npz (boards/policies/legal_masks/outcomes) as a V2Batch.

    Missing fields are filled with sensible defaults:
    - aux_policy = uniform over legal moves (no opponent data in imitation)
    - wdl_short = wdl_terminal (terminal outcome is the best horizon-8 proxy)
    - mlh = 0 (no game-length info in the imitation format)

    This lets the trainer mix imitation data into the self-play replay buffer
    to anchor the policy to the minimax teacher signal during early training.
    Without this, weak self-play can dilute the bootstrap and the model
    regresses (observed: v2 was WORSE than heuristic despite v1 being strong).
    """
    path = Path(path)
    data = np.load(str(path))
    _check_required_keys(data, _IMITATION_REQUIRED_KEYS, path)

    boards = np.asarray(data["boards"]).astype(np.float32)
    policy = np.asarray(data["policies"]).astype(np.float32)
    legal_mask = np.asarray(data["legal_masks"]).astype(bool)
    wdl = np.asarray(data["outcomes"]).astype(np.float32)
    n = boards.shape[0]

    # Uniform aux over legal moves.
    legal_counts = legal_mask.astype(np.float32).sum(axis=1, keepdims=True).clip(min=1)
    aux_policy = (legal_mask.astype(np.float32) / legal_counts).astype(np.float32)

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
    _check_required_keys(data, _V2_REQUIRED_KEYS, path)

    was_full = np.asarray(data["was_full_search"]).astype(bool)
    idx = np.nonzero(was_full)[0]
    if idx.size == 0:
        raise ValueError(f"no full-search rows in {path}")

    boards = _decode_boards(np.asarray(data["boards"])[idx])
    policy = np.asarray(data["policy"])[idx].astype(np.float32)
    aux_policy = np.asarray(data["policy_aux_opp"])[idx].astype(np.float32)
    legal_mask = np.asarray(data["legal_mask"])[idx].astype(bool)
    wdl_terminal = np.asarray(data["wdl_terminal"])[idx].astype(np.float32)
    wdl_short = np.asarray(data["wdl_short"])[idx].astype(np.float32)
    mlh = np.asarray(data["mlh"])[idx].astype(np.float32)
    ply = np.asarray(data["ply"])[idx].astype(np.int32)

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
