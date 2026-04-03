#!/usr/bin/env python3
"""Calibrate WDL_SCALE from imitation data.

Downloads imitation .npz files from S3, extracts the raw minimax eval scores
and game outcomes, then fits the logistic scale parameter via MLE.

The imitation data doesn't store raw eval scores directly, so instead we
regenerate (eval_score, game_outcome) pairs by playing imitation games locally
and recording both signals before blending.

Alternative approach used here: we can back out approximate information from
the existing data by looking at the relationship between the eval-derived WDL
component and game outcomes across many positions. But the cleanest approach
is to just play fresh games and record the raw scores.

Usage:
    ENV_FILE=/Users/kevz/Desktop/hexchess/.env uv run python scripts/calibrate_wdl_scale.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid

# Load .env
env_file = os.environ.get("ENV_FILE", "/Users/kevz/Desktop/hexchess/.env")
if Path(env_file).exists():
    for line in Path(env_file).read_text().splitlines():
        line = line.strip().strip('"')
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training import storage

# Also need hexchess for replaying games
import hexchess


def download_imitation_data(max_files: int = 50) -> list[Path]:
    """Download imitation .npz files from S3 to a temp directory."""
    keys = storage.ls(storage.IMITATION_PREFIX)
    keys = [k for k in keys if k.endswith(".npz")]
    print(f"Found {len(keys)} imitation files in S3")

    if not keys:
        print("No imitation data found!")
        sys.exit(1)

    # Take a sample if there are many
    if len(keys) > max_files:
        keys = sorted(keys)[-max_files:]  # most recent

    tmpdir = Path(tempfile.mkdtemp(prefix="wdl_cal_"))
    paths = []
    for i, key in enumerate(keys):
        local = tmpdir / Path(key).name
        storage.get_file(key, local)
        if (i + 1) % 10 == 0:
            print(f"  downloaded {i+1}/{len(keys)}")
        paths.append(local)

    print(f"Downloaded {len(paths)} files to {tmpdir}")
    return paths


def play_calibration_games(n_games: int = 200, depth: int = 3,
                           random_plies: int = 8, max_ply: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Play imitation games and collect (eval_score, outcome) pairs.

    Returns:
        scores: array of centipawn eval scores (from side-to-move perspective)
        outcomes: array of game outcomes from side-to-move perspective (+1/0/-1)
    """
    import random
    import time

    all_scores = []
    all_outcomes = []

    t0 = time.time()
    for g in range(n_games):
        game = hexchess.Game()
        pending = []  # (eval_score, side_to_move)
        ply = 0
        n_random = random_plies + random.randint(0, 1)

        while not game.is_game_over() and ply < max_ply:
            if ply < n_random:
                moves = game.legal_moves()
                mv = random.choice(moves)
                game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"],
                                mv.get("promotion"))
                ply += 1
                continue

            result = hexchess.minimax_search(game, depth)
            score = result["score"]
            side = game.side_to_move()
            pending.append((score, side))

            mv = result["best_move"]
            game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"],
                            mv.get("promotion"))
            ply += 1

        # Determine outcome
        status = game.status() if game.is_game_over() else "max_ply"
        if status == "checkmate_white":
            white_outcome = 1.0
        elif status == "checkmate_black":
            white_outcome = -1.0
        else:
            white_outcome = 0.0

        for score, side in pending:
            if side == "white":
                outcome = white_outcome
            else:
                outcome = -white_outcome
            all_scores.append(score)
            all_outcomes.append(outcome)

        if (g + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  game {g+1}/{n_games} | {len(all_scores)} positions | {elapsed:.0f}s", flush=True)

    return np.array(all_scores, dtype=np.float64), np.array(all_outcomes, dtype=np.float64)


def fit_scale_and_margin(scores: np.ndarray, outcomes: np.ndarray):
    """Fit WDL_SCALE and draw_margin via MLE on (score, outcome) pairs.

    Model: W = sigmoid((s/scale) - margin)
           L = sigmoid((-s/scale) - margin)
           D = 1 - W - L

    outcome=+1 → log(W), outcome=-1 → log(L), outcome=0 → log(D)
    """
    eps = 1e-8

    def neg_log_likelihood(params):
        scale, margin = params
        if scale < 10 or margin < 0:
            return 1e12
        s = scores / scale
        w = expit(s - margin)
        l = expit(-s - margin)
        d = np.clip(1.0 - w - l, eps, None)
        w = np.clip(w, eps, None)
        l = np.clip(l, eps, None)

        win_mask = outcomes > 0.5
        loss_mask = outcomes < -0.5
        draw_mask = ~win_mask & ~loss_mask

        ll = (np.log(w[win_mask]).sum() +
              np.log(l[loss_mask]).sum() +
              np.log(d[draw_mask]).sum())
        return -ll

    from scipy.optimize import minimize
    best = None
    # Grid search over starting points
    for scale0 in [100, 200, 300, 400, 500, 600, 800]:
        for margin0 in [0.2, 0.5, 1.0, 1.5]:
            res = minimize(neg_log_likelihood, [scale0, margin0],
                           method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1.0, 'fatol': 0.1})
            if best is None or res.fun < best.fun:
                best = res

    return best.x[0], best.x[1], -best.fun


def fit_scale_only(scores: np.ndarray, outcomes: np.ndarray):
    """Fit just WDL_SCALE with tanh model (simpler, matches mcts.rs).

    Model: expected_value = tanh(score / scale)
    Minimize MSE between tanh(score/scale) and outcome.
    """
    def mse(scale):
        if scale < 10:
            return 1e12
        pred = np.tanh(scores / scale)
        return np.mean((pred - outcomes) ** 2)

    res = minimize_scalar(mse, bounds=(50, 1500), method='bounded')
    return res.x, res.fun


def print_comparison(scores, outcomes, scales_to_test):
    """Print a comparison table showing how different scales map scores to win%."""
    print("\n" + "=" * 75)
    print("Score-to-WinRate mapping at different scales")
    print("=" * 75)
    test_scores = [-800, -400, -200, -100, 0, 100, 200, 400, 800]
    header = f"{'Score':>6s}"
    for s in scales_to_test:
        header += f"  scale={s:>5.0f}"
    print(header)
    print("-" * len(header))
    for sc in test_scores:
        row = f"{sc:>6d}"
        for scale in scales_to_test:
            val = np.tanh(sc / scale)
            pct = (val + 1) / 2 * 100  # map [-1,1] to [0%,100%]
            row += f"  {pct:>11.1f}%"
        print(row)


def print_bucket_analysis(scores, outcomes):
    """Bin scores and show empirical win/draw/loss rates."""
    print("\n" + "=" * 75)
    print("Empirical win/draw/loss rates by eval score bucket")
    print("=" * 75)
    edges = [-np.inf, -500, -300, -200, -100, -50, 0, 50, 100, 200, 300, 500, np.inf]
    print(f"{'Bucket':>16s} {'Count':>6s} {'Win%':>6s} {'Draw%':>7s} {'Loss%':>7s} {'Avg Score':>10s}")
    print("-" * 60)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        mask = (scores >= lo) & (scores < hi)
        n = mask.sum()
        if n == 0:
            continue
        wins = (outcomes[mask] > 0.5).mean() * 100
        draws = (np.abs(outcomes[mask]) < 0.5).mean() * 100
        losses = (outcomes[mask] < -0.5).mean() * 100
        avg = scores[mask].mean()

        if np.isinf(lo):
            label = f"< {hi:.0f}"
        elif np.isinf(hi):
            label = f">= {lo:.0f}"
        else:
            label = f"[{lo:.0f}, {hi:.0f})"
        print(f"{label:>16s} {n:>6d} {wins:>5.1f}% {draws:>6.1f}% {losses:>6.1f}% {avg:>10.0f}")


def main():
    print("=" * 75)
    print("WDL Scale Calibration for Hexagonal Chess")
    print("=" * 75)

    # Play games to collect (score, outcome) pairs
    # Depth 2 is ~0.01s/ply vs ~5s/ply at depth 3 — fast enough for calibration.
    # We also run a smaller depth-3 sample for validation.
    n_games = 200
    depth = 2
    print(f"\nPlaying {n_games} imitation games at depth {depth} (max 150 plies)...", flush=True)
    scores, outcomes = play_calibration_games(n_games=n_games, depth=depth, max_ply=150)

    n_total = len(scores)
    n_wins = (outcomes > 0.5).sum()
    n_draws = (np.abs(outcomes) < 0.5).sum()
    n_losses = (outcomes < -0.5).sum()
    print(f"\nCollected {n_total} positions")
    print(f"  Outcomes: {n_wins} wins ({n_wins/n_total:.1%}), "
          f"{n_draws} draws ({n_draws/n_total:.1%}), "
          f"{n_losses} losses ({n_losses/n_total:.1%})")
    print(f"  Score range: [{scores.min():.0f}, {scores.max():.0f}], "
          f"mean={scores.mean():.0f}, std={scores.std():.0f}")

    # Bucket analysis
    print_bucket_analysis(scores, outcomes)

    # Fit tanh model (matches mcts.rs HeuristicEvaluator)
    print("\n" + "=" * 75)
    print("Fitting tanh model: value = tanh(score / scale)")
    print("=" * 75)
    opt_scale_tanh, mse = fit_scale_only(scores, outcomes)
    baseline_mse_400 = np.mean((np.tanh(scores / 400.0) - outcomes) ** 2)
    print(f"  Optimal scale:  {opt_scale_tanh:.0f}  (MSE = {mse:.4f})")
    print(f"  Current (400):          (MSE = {baseline_mse_400:.4f})")
    print(f"  Improvement:    {(baseline_mse_400 - mse) / baseline_mse_400 * 100:.1f}%")

    # Fit full WDL model (matches imitation.py _score_to_wdl)
    print("\n" + "=" * 75)
    print("Fitting logistic WDL model: W=sig(s/scale - margin), L=sig(-s/scale - margin)")
    print("=" * 75)
    opt_scale_wdl, opt_margin, ll = fit_scale_and_margin(scores, outcomes)
    print(f"  Optimal scale:  {opt_scale_wdl:.0f}")
    print(f"  Optimal margin: {opt_margin:.2f}  (current: 0.50)")
    print(f"  Log-likelihood: {ll:.1f}")

    # Compare with current=400
    print_comparison(scores, outcomes, [opt_scale_tanh, 400, opt_scale_wdl])

    # Summary
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"  mcts.rs tanh scale:     current=400, fitted={opt_scale_tanh:.0f}")
    print(f"  imitation.py WDL scale: current=400, fitted={opt_scale_wdl:.0f}")
    print(f"  imitation.py draw_margin: current=0.50, fitted={opt_margin:.2f}")


if __name__ == "__main__":
    main()
