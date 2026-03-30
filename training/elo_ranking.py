"""Elo ranking: baselines vs recent model versions.

Run independently of the training loop to track strength over time.
Plays round-robin matches between fixed baselines (heuristic MCTS,
minimax depth 2, minimax depth 3) and the N most recent model versions
(from models/vN.onnx snapshots), then computes Elo ratings via MLE.

Usage:
    python -m training elo-ranking                  # default: last 10 versions, 6 games/matchup
    python -m training elo-ranking --versions 5     # last 5 versions
    python -m training elo-ranking --games 10       # 10 games per matchup (5 per side)
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .config import _data_root

try:
    import hexchess
except ImportError:
    hexchess = None

# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------


class Player(Protocol):
    name: str

    def pick_move(self, game) -> dict: ...


class MinimaxPlayer:
    def __init__(self, name: str, depth: int):
        self.name = name
        self.depth = depth

    def pick_move(self, game):
        return hexchess.minimax_search(game, self.depth)["best_move"]


class MctsPlayer:
    def __init__(self, name: str, simulations: int, model_path: str | None = None):
        self.name = name
        self.search = hexchess.MctsSearch(
            simulations=simulations,
            model_path=model_path,
        )

    def pick_move(self, game):
        return self.search.run(game, temperature=0.0)["best_move"]


def _baselines(simulations: int = 500) -> list[Player]:
    return [
        MinimaxPlayer(name="Minimax-2", depth=2),
        MinimaxPlayer(name="Minimax-3", depth=3),
        MctsPlayer(name="Heuristic", simulations=simulations),
    ]


def _model_players(num_versions: int, simulations: int = 500) -> list[Player]:
    """Build players from versioned model snapshots (models/vN.onnx)."""
    models_dir = _data_root() / "models"
    if not models_dir.exists():
        return []

    versions = []
    for f in models_dir.glob("v*.onnx"):
        try:
            v = int(f.stem[1:])  # "v12" -> 12
            versions.append((v, f))
        except ValueError:
            continue

    versions.sort(key=lambda x: x[0])
    selected = versions[-num_versions:]
    return [
        MctsPlayer(
            name=f"v{v}",
            simulations=simulations,
            model_path=str(path),
        )
        for v, path in selected
    ]


# ---------------------------------------------------------------------------
# Match play
# ---------------------------------------------------------------------------


def play_game(white: Player, black: Player, max_moves: int = 300) -> str:
    """Play one game. Returns 'white', 'black', or 'draw'."""
    game = hexchess.Game()
    move_count = 0

    while not game.is_game_over() and move_count < max_moves:
        player = white if game.side_to_move() == "white" else black
        mv = player.pick_move(game)
        game.apply_move(
            mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion")
        )
        move_count += 1

    status = game.status()
    if status == "checkmate_white":
        return "white"
    elif status == "checkmate_black":
        return "black"
    return "draw"


def play_matchup(
    a: Player, b: Player, games_per_side: int = 3
) -> dict:
    """Play a matchup (games_per_side with each color). Returns result dict."""
    a_wins, b_wins, draws = 0, 0, 0
    total = games_per_side * 2

    for i in range(total):
        if i < games_per_side:
            white, black = a, b
        else:
            white, black = b, a

        outcome = play_game(white, black)

        if outcome == "draw":
            draws += 1
        elif (outcome == "white" and white is a) or (outcome == "black" and black is a):
            a_wins += 1
        else:
            b_wins += 1

    return {"a": a.name, "b": b.name, "a_wins": a_wins, "b_wins": b_wins, "draws": draws}


# ---------------------------------------------------------------------------
# Elo computation (maximum likelihood)
# ---------------------------------------------------------------------------


def compute_elo(
    players: list[str],
    results: list[dict],
    anchor: str = "Minimax-2",
    anchor_elo: float = 0.0,
    iterations: int = 100,
) -> dict[str, float]:
    """Compute Elo ratings from pairwise results via iterative MLE.

    Anchors one player at a fixed rating to set the scale.
    """
    elo = {p: 1500.0 for p in players}
    if anchor in elo:
        elo[anchor] = anchor_elo

    scores: dict[tuple[str, str], tuple[float, int]] = {}
    for r in results:
        a, b = r["a"], r["b"]
        total = r["a_wins"] + r["b_wins"] + r["draws"]
        if total == 0:
            continue
        a_score = r["a_wins"] + r["draws"] * 0.5
        b_score = r["b_wins"] + r["draws"] * 0.5
        scores[(a, b)] = (a_score, total)
        scores[(b, a)] = (b_score, total)

    for _ in range(iterations):
        for p in players:
            if p == anchor:
                continue
            actual_total = 0.0
            expected_total = 0.0
            games_total = 0

            for opp in players:
                if opp == p:
                    continue
                key = (p, opp)
                if key not in scores:
                    continue
                actual, n_games = scores[key]
                expected_score = n_games / (1 + 10 ** ((elo[opp] - elo[p]) / 400))
                actual_total += actual
                expected_total += expected_score
                games_total += n_games

            if games_total == 0 or expected_total == 0:
                continue
            ratio = max(0.1, min(10.0, actual_total / expected_total))
            elo[p] += 400 * math.log10(ratio) * 0.5  # damped update

    # Re-anchor
    if anchor in elo:
        offset = anchor_elo - elo[anchor]
        for p in elo:
            elo[p] += offset

    return {p: round(elo[p]) for p in players}


def format_elo_table(ratings: dict[str, int | float]) -> str:
    """Format Elo ratings as a ranked table string."""
    lines = []
    sorted_players = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, elo) in enumerate(sorted_players, 1):
        lines.append(f"  {rank}. {name:<20s} {elo:>+6d}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_elo_ranking(
    num_versions: int = 10,
    games_per_side: int = 3,
    simulations: int = 500,
) -> dict:
    """Run full Elo ranking and return results.

    Returns a dict with 'ratings', 'matchups', and 'timestamp'.
    """
    if hexchess is None:
        raise ImportError("hexchess Python bindings not available")

    baselines = _baselines(simulations)
    models = _model_players(num_versions, simulations)

    if not models:
        print("No model versions found. Run training first.")
        return {}

    all_players = baselines + models
    player_names = [p.name for p in all_players]

    print(f"Elo ranking: {len(all_players)} players, {games_per_side} games/side")
    for p in all_players:
        print(f"  - {p.name}")

    t0 = time.time()
    all_results = []

    matchups = []
    for i in range(len(all_players)):
        for j in range(i + 1, len(all_players)):
            matchups.append((all_players[i], all_players[j]))

    total_matchups = len(matchups)
    for idx, (a, b) in enumerate(matchups):
        print(f"\n[{idx + 1}/{total_matchups}] {a.name} vs {b.name}...", flush=True)
        result = play_matchup(a, b, games_per_side)
        all_results.append(result)
        decided = result["a_wins"] + result["b_wins"]
        rate = result["a_wins"] / decided if decided > 0 else 0.5
        print(
            f"  {result['a_wins']}W-{result['b_wins']}L-{result['draws']}D "
            f"({rate:.0%} for {a.name})"
        )

    elapsed = time.time() - t0

    ratings = compute_elo(player_names, all_results)

    print(f"\n{'='*50}")
    print("  ELO RANKINGS")
    print(f"{'='*50}")
    print(format_elo_table(ratings))
    print(f"\n  ({elapsed:.0f}s elapsed, {games_per_side * 2} games per matchup)")

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "games_per_side": games_per_side,
        "simulations": simulations,
        "ratings": ratings,
        "matchups": all_results,
    }

    # Notify Slack
    from .slack import notify_elo_ranking
    notify_elo_ranking(ratings, elapsed)

    results_path = _data_root() / "elo_rankings.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a") as f:
        f.write(json.dumps(output) + "\n")
    print(f"\n  Results appended to {results_path}")

    return output
