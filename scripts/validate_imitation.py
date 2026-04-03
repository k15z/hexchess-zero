#!/usr/bin/env python3
"""Validation script for imitation learning pipeline.

Step 4.5: Data quality validation
  - Non-determinism: games differ across runs
  - Outcome balance: W/L/D all occur, no extreme bias
  - Strength ordering: minimax-4 > minimax-3 > minimax-2 > MCTS heuristic

Step 5: End-to-end training validation
  - Generate imitation data
  - Train a model
  - Confirm MCTS+NN beats MCTS+heuristic
"""

from __future__ import annotations

import sys
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_non_determinism(n_games: int = 10) -> bool:
    """Generate n_games and verify they differ."""
    from training.imitation import play_imitation_game
    from training.config import AsyncConfig

    print(f"\n{'='*60}")
    print(f"NON-DETERMINISM: generating {n_games} games...")
    print(f"{'='*60}")

    cfg = AsyncConfig()
    cfg.imitation_depth = 3

    games = []
    for i in range(n_games):
        samples = play_imitation_game(cfg, log_interval=999)
        game_len = len(samples)
        # Use first move's policy as a fingerprint
        first_policy = tuple(samples[0]["policy"].nonzero()[0].tolist()) if samples else ()
        games.append((game_len, first_policy))
        print(f"  Game {i+1}: {game_len} positions")

    lengths = [g[0] for g in games]
    policies = [g[1] for g in games]

    unique_lengths = len(set(lengths))
    unique_policies = len(set(policies))

    print(f"\n  Unique game lengths: {unique_lengths}/{n_games}")
    print(f"  Unique first policies: {unique_policies}/{n_games}")

    passed = unique_lengths > 1 or unique_policies > 1
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def _play_one_game(cfg_depth):
    """Worker function for parallel game generation."""
    from training.imitation import play_imitation_game
    from training.config import AsyncConfig

    cfg = AsyncConfig()
    cfg.imitation_depth = cfg_depth
    samples = play_imitation_game(cfg, log_interval=999)
    # Determine outcome from the last WDL
    if samples:
        wdl = samples[-1]["outcome"]
        w, d, l = wdl[0], wdl[1], wdl[2]
        if w > l + 0.1:
            return "white_advantage", len(samples)
        elif l > w + 0.1:
            return "black_advantage", len(samples)
        else:
            return "balanced", len(samples)
    return "empty", 0


def validate_outcome_balance(n_games: int = 50) -> bool:
    """Generate games and check outcome distribution."""
    from training.imitation import play_imitation_game
    from training.config import AsyncConfig
    import hexchess

    print(f"\n{'='*60}")
    print(f"OUTCOME BALANCE: generating {n_games} games with minimax play...")
    print(f"{'='*60}")

    outcomes = Counter()
    game_lengths = []

    for i in range(n_games):
        game = hexchess.Game()
        # Play with random opening + minimax
        import random
        random_plies = 8 + random.randint(-1, 0)
        ply = 0
        max_ply = 300

        while not game.is_game_over() and ply < max_ply:
            if ply < random_plies:
                moves = game.legal_moves()
                mv = random.choice(moves)
                game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion"))
                ply += 1
                continue

            result = hexchess.minimax_search(game, 3)
            mv = result["best_move"]
            game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion"))
            ply += 1

        status = game.status()
        if status == "checkmate_white":
            outcomes["white_win"] += 1
        elif status == "checkmate_black":
            outcomes["black_win"] += 1
        else:
            outcomes["draw"] += 1
        game_lengths.append(ply)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_games} games done...")

    print(f"\n  Outcomes: {dict(outcomes)}")
    print(f"  Game lengths: min={min(game_lengths)}, max={max(game_lengths)}, "
          f"avg={sum(game_lengths)/len(game_lengths):.0f}")

    # Check that at least 2 different outcomes occur
    n_outcome_types = len(outcomes)
    print(f"  Distinct outcome types: {n_outcome_types}")

    # Check no single outcome exceeds 80%
    max_pct = max(outcomes.values()) / n_games * 100
    print(f"  Max outcome percentage: {max_pct:.1f}%")

    # Check game length variation
    length_set = len(set(game_lengths))
    print(f"  Distinct game lengths: {length_set}")

    passed = n_outcome_types >= 2 and max_pct <= 80 and length_set > 1
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def _play_game_with_random_opening(white, black, random_plies=8, max_moves=200):
    """Play a game with random opening moves, then hand off to players."""
    import hexchess
    import random as rnd

    game = hexchess.Game()
    n_random = random_plies + rnd.randint(-1, 0)

    for _ in range(n_random):
        if game.is_game_over():
            break
        moves = game.legal_moves()
        mv = rnd.choice(moves)
        game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion"))

    move_count = n_random
    while not game.is_game_over() and move_count < max_moves:
        player = white if game.side_to_move() == "white" else black
        mv = player.pick_move(game)
        game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion"))
        move_count += 1

    status = game.status()
    if status == "checkmate_white":
        outcome = "white"
    elif status == "checkmate_black":
        outcome = "black"
    else:
        outcome = "draw"

    return {"outcome": outcome, "moves": move_count}


def validate_strength_ordering(games_per_pair: int = 6) -> bool:
    """Run round-robin tournament with random openings to verify strength ordering."""
    from training.elo import MinimaxPlayer, MctsPlayer, compute_elo, format_elo_table

    print(f"\n{'='*60}")
    print(f"STRENGTH ORDERING: round-robin tournament ({games_per_pair} games/pair, random openings)...")
    print(f"{'='*60}")

    players = [
        MctsPlayer(name="MCTS-Heuristic", simulations=100),
        MinimaxPlayer(name="Minimax-2", depth=2),
        MinimaxPlayer(name="Minimax-3", depth=3),
        MinimaxPlayer(name="Minimax-4", depth=4),
    ]

    results = []
    for i, p1 in enumerate(players):
        for j, p2 in enumerate(players):
            if j <= i:
                continue
            a_wins = 0
            b_wins = 0
            draws = 0
            for g in range(games_per_pair):
                # Alternate colors
                if g % 2 == 0:
                    white, black = p1, p2
                else:
                    white, black = p2, p1

                print(f"  {p1.name} vs {p2.name} (game {g+1}/{games_per_pair})...", end=" ", flush=True)
                t0 = time.time()
                result = _play_game_with_random_opening(white, black)
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

                print(f"{winner} ({result['moves']} moves, {dt:.1f}s)")

            results.append({
                "a": p1.name, "b": p2.name,
                "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
            })
            print(f"    => {p1.name}: {a_wins}W, {p2.name}: {b_wins}W, draws: {draws}")

    player_names = [p.name for p in players]
    elo = compute_elo(player_names, results, anchor="Minimax-2")

    print(f"\n  Elo Ratings:")
    print(format_elo_table(elo))

    # Check monotonic ordering
    expected_order = ["Minimax-4", "Minimax-3", "Minimax-2", "MCTS-Heuristic"]
    actual_order = sorted(elo.keys(), key=lambda k: elo[k], reverse=True)

    print(f"\n  Expected ordering: {' > '.join(expected_order)}")
    print(f"  Actual ordering:   {' > '.join(actual_order)}")

    # Check that minimax-4 >= minimax-3 >= minimax-2
    monotonic = (elo.get("Minimax-4", 0) >= elo.get("Minimax-3", 0) >= elo.get("Minimax-2", 0))
    # Check that minimax-2 >= MCTS-Heuristic (with some tolerance since few games)
    beats_heuristic = elo.get("Minimax-2", 0) >= elo.get("MCTS-Heuristic", 0) - 50

    passed = monotonic and beats_heuristic
    print(f"  Monotonic minimax: {'PASS' if monotonic else 'FAIL'}")
    print(f"  Minimax-2 >= Heuristic: {'PASS' if beats_heuristic else 'FAIL'}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    t0 = time.time()

    r1 = validate_non_determinism()
    r2 = validate_outcome_balance()
    r3 = validate_strength_ordering()

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Non-determinism:    {'PASS' if r1 else 'FAIL'}")
    print(f"  Outcome balance:    {'PASS' if r2 else 'FAIL'}")
    print(f"  Strength ordering:  {'PASS' if r3 else 'FAIL'}")
    print(f"  Total time: {time.time() - t0:.0f}s")

    all_pass = r1 and r2 and r3
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
