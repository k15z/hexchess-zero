#!/usr/bin/env python3
"""Ablation study for positional evaluation signals.

Measures two things for each eval weight configuration:
1. Policy sharpness — how many unique scores and how concentrated the top moves are
2. Playing strength — tournament vs material-only baseline at multiple depths

Usage:
    uv run python scripts/eval_ablation.py
"""

from __future__ import annotations

import random
import time

import numpy as np

import hexchess


# ---------------------------------------------------------------------------
# Configs to test
# ---------------------------------------------------------------------------

CONFIGS: dict[str, hexchess.EvalWeights] = {
    "material_only": hexchess.EvalWeights.material_only(),
    "all_signals": hexchess.EvalWeights(),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_positions(n: int = 50, random_plies: int = 10) -> list:
    positions = []
    for _ in range(n):
        game = hexchess.Game()
        plies = random_plies + random.randint(-2, 2)
        for _ in range(plies):
            if game.is_game_over():
                break
            moves = game.legal_moves()
            mv = random.choice(moves)
            game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"],
                            mv.get("promotion"))
        if not game.is_game_over():
            positions.append(game)
    return positions


def measure_sharpness(weights: hexchess.EvalWeights, positions: list,
                      depth: int = 3, temperature: float = 100.0) -> dict:
    unique_scores_list = []
    top3_masses = []
    top1_masses = []

    for game in positions:
        result = hexchess.minimax_search_with_policy(game, depth, weights=weights)
        scores = [m["score"] for m in result["moves"]]
        unique_scores_list.append(len(set(scores)))

        scores_arr = np.array(scores, dtype=np.float64)
        scaled = scores_arr / temperature
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        sorted_probs = np.sort(probs)[::-1]

        top1_masses.append(float(sorted_probs[0]))
        top3_masses.append(float(sorted_probs[:3].sum()))

    return {
        "unique_scores": np.mean(unique_scores_list),
        "top1_mass": np.mean(top1_masses),
        "top3_mass": np.mean(top3_masses),
    }


def play_match(w_a: hexchess.EvalWeights, w_b: hexchess.EvalWeights,
               depth: int = 3, games: int = 10, random_plies: int = 8) -> dict:
    a_wins = b_wins = draws = 0

    for g in range(games):
        game = hexchess.Game()
        n_random = random_plies + random.randint(-1, 0)
        for _ in range(n_random):
            if game.is_game_over():
                break
            moves = game.legal_moves()
            mv = random.choice(moves)
            game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"],
                            mv.get("promotion"))

        if g % 2 == 0:
            white_w, black_w = w_a, w_b
            a_is_white = True
        else:
            white_w, black_w = w_b, w_a
            a_is_white = False

        move_count = 0
        while not game.is_game_over() and move_count < 200:
            w = white_w if game.side_to_move() == "white" else black_w
            result = hexchess.minimax_search(game, depth, weights=w)
            mv = result["best_move"]
            game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"],
                            mv.get("promotion"))
            move_count += 1

        status = game.status()
        if status == "checkmate_white":
            if a_is_white:
                a_wins += 1
            else:
                b_wins += 1
        elif status == "checkmate_black":
            if a_is_white:
                b_wins += 1
            else:
                a_wins += 1
        else:
            draws += 1

    return {"a_wins": a_wins, "b_wins": b_wins, "draws": draws}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    # Part A: Sharpness
    print("=" * 70)
    print("PART A: Policy Sharpness (depth=3, 50 positions, temp=100)")
    print("=" * 70)

    positions = generate_positions(50)
    print(f"Generated {len(positions)} positions\n")

    print(f"{'Config':<22s} {'Unique Scores':>14s} {'Top-1 Mass':>11s} {'Top-3 Mass':>11s}")
    print("-" * 60)

    for name, weights in CONFIGS.items():
        result = measure_sharpness(weights, positions, depth=3, temperature=100.0)
        print(f"{name:<22s} {result['unique_scores']:>14.1f} {result['top1_mass']:>11.3f} {result['top3_mass']:>11.3f}")

    # Part B: Multi-depth tournament
    print(f"\n{'=' * 70}")
    print("PART B: all_signals vs material_only at multiple depths")
    print("=" * 70)

    new_w = CONFIGS["all_signals"]
    old_w = CONFIGS["material_only"]

    for depth, games in [(2, 10), (3, 10), (4, 4)]:
        print(f"\n  Depth {depth} ({games} games)...", flush=True)
        dt0 = time.time()
        result = play_match(new_w, old_w, depth=depth, games=games)
        dt = time.time() - dt0
        total = result["a_wins"] + result["b_wins"] + result["draws"]
        score = (result["a_wins"] + 0.5 * result["draws"]) / total if total > 0 else 0
        print(f"    New: {result['a_wins']}W  Old: {result['b_wins']}W  Draws: {result['draws']}D  "
              f"Score: {score:.0%}  ({dt:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
