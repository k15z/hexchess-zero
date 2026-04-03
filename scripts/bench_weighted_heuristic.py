#!/usr/bin/env python3
"""Benchmark: MCTS weighted heuristic vs baselines + color bias check.

Usage:
    uv run python scripts/bench_weighted_heuristic.py [games_per_side]
"""

from __future__ import annotations

import random
import sys
import time

import hexchess


class MctsPlayer:
    def __init__(self, name: str, simulations: int, use_weighted_eval: bool = False):
        self.name = name
        self.search = hexchess.MctsSearch(
            simulations=simulations,
            use_weighted_eval=use_weighted_eval,
        )

    def pick_move(self, game):
        return self.search.run(game, temperature=0.0)["best_move"]


class MinimaxPlayer:
    def __init__(self, name: str, depth: int):
        self.name = name
        self.depth = depth

    def pick_move(self, game):
        return hexchess.minimax_search(game, self.depth)["best_move"]


def play_game(white, black, max_moves: int = 300, random_plies: int = 8) -> dict:
    game = hexchess.Game()
    n_random = random_plies + random.randint(-1, 0)
    for _ in range(n_random):
        if game.is_game_over():
            break
        moves = game.legal_moves()
        mv = random.choice(moves)
        game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion"))

    move_count = 0
    white_time = 0.0
    black_time = 0.0
    white_moves = 0
    black_moves = 0

    while not game.is_game_over() and move_count < max_moves:
        is_white = game.side_to_move() == "white"
        player = white if is_white else black
        t0 = time.monotonic()
        mv = player.pick_move(game)
        dt = time.monotonic() - t0
        if is_white:
            white_time += dt
            white_moves += 1
        else:
            black_time += dt
            black_moves += 1
        game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion"))
        move_count += 1

    status = game.status()
    if status == "checkmate_white":
        outcome = "white"
    elif status == "checkmate_black":
        outcome = "black"
    else:
        outcome = "draw"

    return {
        "outcome": outcome,
        "moves": move_count,
        "white_time": white_time,
        "black_time": black_time,
        "white_moves": white_moves,
        "black_moves": black_moves,
    }


def run_matchup(player_a, player_b, games_per_side: int = 5) -> dict:
    total = games_per_side * 2
    a_wins = b_wins = draws = 0
    a_total_time = 0.0
    b_total_time = 0.0
    a_total_moves = 0
    b_total_moves = 0

    print(f"\n  {player_a.name}  vs  {player_b.name}  ({total} games)", flush=True)

    for i in range(total):
        if i < games_per_side:
            white, black = player_a, player_b
            a_is_white = True
        else:
            white, black = player_b, player_a
            a_is_white = False

        r = play_game(white, black)

        if a_is_white:
            a_total_time += r["white_time"]
            a_total_moves += r["white_moves"]
            b_total_time += r["black_time"]
            b_total_moves += r["black_moves"]
        else:
            a_total_time += r["black_time"]
            a_total_moves += r["black_moves"]
            b_total_time += r["white_time"]
            b_total_moves += r["white_moves"]

        if r["outcome"] == "draw":
            draws += 1
            tag = "D"
        elif (r["outcome"] == "white" and a_is_white) or (
            r["outcome"] == "black" and not a_is_white
        ):
            a_wins += 1
            tag = "A"
        else:
            b_wins += 1
            tag = "B"

        print(f"    {i+1:>2}/{total} [{tag}] {r['moves']}mv {r['white_time']+r['black_time']:.1f}s", flush=True)

    a_score = a_wins + draws * 0.5
    a_rate = a_score / total * 100
    a_ms = (a_total_time / a_total_moves * 1000) if a_total_moves else 0
    b_ms = (b_total_time / b_total_moves * 1000) if b_total_moves else 0

    print(f"  => {player_a.name}: {a_wins}W-{b_wins}L-{draws}D ({a_rate:.0f}%)  "
          f"ms/mv: {a_ms:.1f} vs {b_ms:.1f}", flush=True)

    return {
        "a": player_a.name, "b": player_b.name,
        "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "a_rate": a_rate, "a_ms": a_ms, "b_ms": b_ms,
    }


def run_self_play(player_factory, n_games: int = 30) -> dict:
    """Play a player against itself to check for color bias."""
    white_wins = black_wins = draws = 0

    print(f"\n  Self-play color bias: {player_factory().name} ({n_games} games)", flush=True)

    for i in range(n_games):
        # Fresh instances each game to avoid TT leaking between sides
        white = player_factory()
        black = player_factory()
        r = play_game(white, black)

        if r["outcome"] == "white":
            white_wins += 1
            tag = "W"
        elif r["outcome"] == "black":
            black_wins += 1
            tag = "B"
        else:
            draws += 1
            tag = "D"

        print(f"    {i+1:>2}/{n_games} [{tag}] {r['moves']}mv", flush=True)

    print(f"  => White: {white_wins}  Black: {black_wins}  Draw: {draws}", flush=True)
    return {"white_wins": white_wins, "black_wins": black_wins, "draws": draws}


def main():
    games_per_side = 10
    if len(sys.argv) > 1:
        games_per_side = int(sys.argv[1])

    sims = 500

    print(f"=== PART 1: Round-robin tournament ({games_per_side} games/side, {sims} sims) ===")

    players = [
        MctsPlayer(f"MCTS-New({sims}s)", sims, use_weighted_eval=True),
        MctsPlayer(f"MCTS-Old({sims}s)", sims, use_weighted_eval=False),
        MinimaxPlayer("Minimax-2", depth=2),
        MinimaxPlayer("Minimax-3", depth=3),
    ]

    all_results = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            result = run_matchup(players[i], players[j], games_per_side)
            all_results.append(result)

    print(f"\n{'=' * 72}")
    print("  TOURNAMENT RESULTS")
    print(f"{'=' * 72}")
    print(f"  {'Matchup':<40s} {'W-L-D':>10s} {'Win%':>6s} {'ms/mv A':>9s} {'ms/mv B':>9s}")
    print(f"  {'-' * 72}")
    for r in all_results:
        matchup = f"{r['a']} vs {r['b']}"
        wld = f"{r['a_wins']}-{r['b_wins']}-{r['draws']}"
        print(f"  {matchup:<40s} {wld:>10s} {r['a_rate']:>5.0f}% {r['a_ms']:>8.1f} {r['b_ms']:>8.1f}")

    print(f"\n=== PART 2: Color bias check (MCTS-New self-play, 30 games) ===")
    bias = run_self_play(
        lambda: MctsPlayer(f"MCTS-New({sims}s)", sims, use_weighted_eval=True),
        n_games=30,
    )

    print(f"\n{'=' * 72}")
    print("  COLOR BIAS RESULTS")
    print(f"{'=' * 72}")
    total = bias["white_wins"] + bias["black_wins"] + bias["draws"]
    print(f"  White wins: {bias['white_wins']}/{total} ({bias['white_wins']/total*100:.0f}%)")
    print(f"  Black wins: {bias['black_wins']}/{total} ({bias['black_wins']/total*100:.0f}%)")
    print(f"  Draws:      {bias['draws']}/{total} ({bias['draws']/total*100:.0f}%)")


if __name__ == "__main__":
    main()
