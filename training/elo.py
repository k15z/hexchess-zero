"""Shared Elo types, game play, and rating computation."""

from __future__ import annotations

import math
from typing import Protocol

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


def baselines(simulations: int = 500) -> list[Player]:
    return [
        MinimaxPlayer(name="Minimax-2", depth=2),
        MinimaxPlayer(name="Minimax-3", depth=3),
        MinimaxPlayer(name="Minimax-4", depth=4),
        MctsPlayer(name="Heuristic", simulations=simulations),
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


# ---------------------------------------------------------------------------
# Elo computation (maximum likelihood)
# ---------------------------------------------------------------------------


def compute_elo(
    players: list[str],
    results: list[dict],
    anchor: str = "Minimax-2",
    anchor_elo: float = 1500.0,
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
