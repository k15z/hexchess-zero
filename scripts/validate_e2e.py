#!/usr/bin/env python3
"""Step 5: End-to-end training validation.

Generates imitation data, trains a small model, and verifies MCTS+NN
beats MCTS+heuristic. Uses S3 for data storage (same as production pipeline)
but with reduced thresholds for quick local validation.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from loguru import logger  # noqa: E402

from training import storage  # noqa: E402
from training.config import AsyncConfig  # noqa: E402
from training.imitation import play_imitation_game  # noqa: E402
from training.losses import LossWeights, compute_losses  # noqa: E402
from training.model import build_model  # noqa: E402
from training.export import export_to_onnx  # noqa: E402
from training.trainer_loop import ImitationBuffer  # noqa: E402


# ---------------------------------------------------------------------------
# Config overrides for quick local validation
# ---------------------------------------------------------------------------

NUM_WORKERS = max(1, os.cpu_count() or 1)
GAMES_PER_FLUSH = 5


def _collate_samples(samples: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.stack([s[k] for s in samples]) for k in samples[0]}


def generate_imitation_data(cfg: AsyncConfig) -> int:
    """Generate imitation data and upload to S3. Returns total positions."""
    target_positions = cfg.min_positions_to_start
    existing = storage.count_positions(storage.IMITATION_PREFIX)
    if existing >= target_positions:
        logger.info("Already have {:,} positions (target {:,}), skipping generation",
                    existing, target_positions)
        return existing

    needed = target_positions - existing
    logger.info("Have {:,} positions, need {:,} more (target {:,}, {} workers)",
                existing, needed, target_positions, NUM_WORKERS)

    total_games = 0
    total_positions = 0
    pending_samples = []
    pending_games = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(play_imitation_game, cfg) for _ in range(NUM_WORKERS * 2)}

        while total_positions < needed:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                samples = future.result()
                pending_samples.extend(samples)
                pending_games += 1
                total_games += 1
                total_positions += len(samples)

                futures.add(pool.submit(play_imitation_game, cfg))

                if pending_games >= GAMES_PER_FLUSH:
                    key = storage.flush_samples(pending_samples, storage.IMITATION_PREFIX)
                    elapsed = time.time() - t0
                    logger.info("  Flushed {} games, {} pos ({:,}/{:,} new, {:,} total, {:.0f}s) | {}",
                                pending_games, len(pending_samples),
                                total_positions, needed, existing + total_positions, elapsed, key)
                    pending_samples = []
                    pending_games = 0

        for f in futures:
            f.cancel()

    if pending_samples:
        storage.flush_samples(pending_samples, storage.IMITATION_PREFIX)

    elapsed = time.time() - t0
    logger.info("Data generation complete: {} games, {:,} new positions in {:.0f}s ({:,} total)",
                total_games, total_positions, elapsed, existing + total_positions)
    return existing + total_positions


def train_model(cfg: AsyncConfig) -> str:
    """Train a model on imitation data. Returns path to ONNX model."""
    train_steps = cfg.bootstrap_steps
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Training on device: {} for {:,} steps...", device, train_steps)

    model = build_model(cfg).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.bootstrap_learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.l2_regularization,
    )
    loss_weights = LossWeights()

    dataset = ImitationBuffer(
        cfg.data_cache_dir / "imitation",
        max_positions=cfg.min_positions_to_start,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, num_workers=0, collate_fn=_collate_samples
    )

    logger.info("Replay buffer: {} files, {:,} positions",
                len(dataset.files), dataset.total_positions)

    model.train()
    step = 0
    cum_policy_loss = 0.0
    cum_value_loss = 0.0
    t0 = time.time()

    while step < train_steps:
        for batch_np in dataloader:
            if step >= train_steps:
                break

            boards = torch.from_numpy(batch_np["boards"]).to(device)
            legal_mask = torch.from_numpy(batch_np["legal_mask"]).to(device).bool()
            targets = {
                "policy": torch.from_numpy(batch_np["policy"]).to(device),
                "wdl": torch.from_numpy(batch_np["wdl_terminal"]).to(device),
                "mlh": torch.from_numpy(batch_np["mlh"]).to(device),
                "stv": torch.from_numpy(batch_np["wdl_short"]).to(device),
                "aux_policy": torch.from_numpy(batch_np["aux_policy"]).to(device),
            }

            preds = model(boards)
            breakdown = compute_losses(
                preds, targets, legal_mask=legal_mask, weights=loss_weights
            )

            optimizer.zero_grad()
            breakdown.total.backward()
            optimizer.step()

            cum_policy_loss += breakdown.policy.item()
            cum_value_loss += breakdown.value.item()
            step += 1

            if step % 500 == 0 or step == train_steps:
                elapsed = time.time() - t0
                avg_p = cum_policy_loss / step
                avg_v = cum_value_loss / step
                logger.info(
                    "  step {:>5}/{:,} | loss: policy={:.4f} value={:.4f} "
                    "total={:.4f} | {:.1f} steps/s",
                    step, train_steps, avg_p, avg_v, avg_p + avg_v,
                    step / elapsed,
                )

    elapsed = time.time() - t0
    logger.info("Training complete: {:,} steps in {:.0f}s", step, elapsed)

    # Export to ONNX
    cfg.ensure_cache_dirs()
    local_pt = cfg.model_cache_dir / "validate_checkpoint.pt"
    local_onnx = cfg.model_cache_dir / "validate_model.onnx"
    torch.save(model.state_dict(), local_pt)
    export_to_onnx(local_pt, local_onnx, cfg)

    return str(local_onnx)


def evaluate_model(onnx_path: str, cfg: AsyncConfig, games_per_pair: int = 6) -> bool:
    """Play MCTS+NN vs MCTS+heuristic and Minimax-2. Returns True if NN wins."""
    from training.elo import (
        MctsPlayer, MinimaxPlayer, play_game, compute_elo, conservative_ratings,
        format_elo_table,
    )

    logger.info("Evaluating MCTS+NN vs baselines ({} games/pair, {} sims, random openings)...",
                games_per_pair, cfg.num_simulations)

    players = [
        MctsPlayer(name="MCTS-Heuristic", simulations=cfg.num_simulations),
        MctsPlayer(name="MCTS-NN", simulations=cfg.num_simulations, model_path=onnx_path),
        MinimaxPlayer(name="Minimax-2", depth=2),
    ]

    results = []
    for i, p1 in enumerate(players):
        for j, p2 in enumerate(players):
            if j <= i:
                continue
            a_wins = b_wins = draws = 0
            for g in range(games_per_pair):
                white, black = (p1, p2) if g % 2 == 0 else (p2, p1)
                print(f"  {p1.name} vs {p2.name} (game {g+1}/{games_per_pair})...", end=" ", flush=True)
                t0 = time.time()
                result = play_game(white, black, max_moves=200, random_opening_plies=8)
                dt = time.time() - t0

                if result["outcome"] == "white":
                    winner = white.name
                    if white.name == p1.name:
                        a_wins += 1
                    else:
                        b_wins += 1
                elif result["outcome"] == "black":
                    winner = black.name
                    if black.name == p1.name:
                        a_wins += 1
                    else:
                        b_wins += 1
                else:
                    winner = "draw"
                    draws += 1
                print(f"{winner} ({dt:.1f}s)")

            results.append({"a": p1.name, "b": p2.name, "a_wins": a_wins, "b_wins": b_wins, "draws": draws})
            print(f"    => {p1.name}: {a_wins}W, {p2.name}: {b_wins}W, draws: {draws}")

    player_names = [p.name for p in players]
    ratings = compute_elo(player_names, results)

    print("\n  Ratings:")
    print(format_elo_table(ratings))

    scores = conservative_ratings(ratings)
    nn_score = scores.get("MCTS-NN", 0.0)
    heuristic_score = scores.get("MCTS-Heuristic", 0.0)
    nn_beats_heuristic = nn_score > heuristic_score

    print(f"\n  MCTS-NN ({nn_score:+.2f}) vs MCTS-Heuristic ({heuristic_score:+.2f}): "
          f"{'PASS' if nn_beats_heuristic else 'FAIL'}")
    return nn_beats_heuristic


if __name__ == "__main__":
    t0 = time.time()
    cfg = AsyncConfig()

    # Step 1: Generate imitation data
    print(f"\n{'='*60}")
    print("STEP 1: Generate imitation data")
    print(f"{'='*60}")
    total_pos = generate_imitation_data(cfg)

    # Step 2: Train model
    print(f"\n{'='*60}")
    print("STEP 2: Train model")
    print(f"{'='*60}")
    onnx_path = train_model(cfg)

    # Step 3: Evaluate
    print(f"\n{'='*60}")
    print("STEP 3: Evaluate MCTS+NN vs baselines")
    print(f"{'='*60}")
    passed = evaluate_model(onnx_path, cfg)

    print(f"\n{'='*60}")
    print(f"E2E VALIDATION: {'PASS' if passed else 'FAIL'}")
    print(f"Total time: {time.time() - t0:.0f}s")
    print(f"{'='*60}")

    sys.exit(0 if passed else 1)
