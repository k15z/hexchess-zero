#!/usr/bin/env python3
"""Round-robin tournament between different hexchess AI players."""

import sys
import time
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, ".")
import hexchess


@dataclass
class MatchResult:
    white_name: str
    black_name: str
    outcome: str  # "white", "black", "draw"
    moves: int
    seconds: float


class MinimaxPlayer:
    def __init__(self, depth: int):
        self.depth = depth
        self.name = f"Minimax(d={depth})"

    def pick_move(self, game):
        result = hexchess.minimax_search(game, self.depth)
        return result["best_move"]


class MctsPlayer:
    def __init__(self, simulations: int, model_path: Optional[str] = None):
        label = "heuristic" if model_path is None else "model"
        self.name = f"MCTS({simulations}s,{label})"
        self.search = hexchess.MctsSearch(
            simulations=simulations, model_path=model_path
        )

    def pick_move(self, game):
        result = self.search.run(game, temperature=0.0)
        return result["best_move"]


def play_game(white_player, black_player, max_moves=300) -> MatchResult:
    game = hexchess.Game()
    t0 = time.time()
    move_count = 0

    while not game.is_game_over() and move_count < max_moves:
        player = white_player if game.side_to_move() == "white" else black_player
        mv = player.pick_move(game)
        game.apply_move(
            mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion")
        )
        move_count += 1

    elapsed = time.time() - t0
    status = game.status()

    if status == "checkmate_white":
        outcome = "white"
    elif status == "checkmate_black":
        outcome = "black"
    else:
        outcome = "draw"

    return MatchResult(white_player.name, black_player.name, outcome, move_count, elapsed)


def run_matchup(player_a, player_b, games_per_side=5):
    """Play games_per_side games with each color assignment."""
    results = []
    total = games_per_side * 2

    print(f"\n{'='*60}")
    print(f"  {player_a.name}  vs  {player_b.name}")
    print(f"  {total} games ({games_per_side} per side)")
    print(f"{'='*60}")

    a_wins = 0
    b_wins = 0
    draws = 0

    for i in range(total):
        if i < games_per_side:
            white, black = player_a, player_b
        else:
            white, black = player_b, player_a

        r = play_game(white, black)
        results.append(r)

        # Determine winner from perspective of player_a vs player_b
        if r.outcome == "draw":
            draws += 1
            winner_str = "draw"
        elif (r.outcome == "white" and white is player_a) or (
            r.outcome == "black" and black is player_a
        ):
            a_wins += 1
            winner_str = player_a.name
        else:
            b_wins += 1
            winner_str = player_b.name

        print(
            f"  game {i+1}/{total}: {white.name}(W) vs {black.name}(B) "
            f"-> {winner_str} ({r.moves} moves, {r.seconds:.1f}s)"
        )

    a_score = a_wins + draws * 0.5
    a_rate = a_score / total * 100

    print(f"\n  Result: {player_a.name} {a_wins}W-{b_wins}L-{draws}D ({a_rate:.0f}%)")
    return {
        "a": player_a.name,
        "b": player_b.name,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_rate": a_rate,
    }


def main():
    games_per_side = 5
    if len(sys.argv) > 1:
        games_per_side = int(sys.argv[1])

    players = [
        MinimaxPlayer(depth=2),
        MinimaxPlayer(depth=3),
        MctsPlayer(simulations=500, model_path=None),
    ]

    print(f"Tournament: {len(players)} players, {games_per_side} games/side")
    for p in players:
        print(f"  - {p.name}")

    all_results = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            result = run_matchup(players[i], players[j], games_per_side)
            all_results.append(result)

    print(f"\n{'='*60}")
    print("  TOURNAMENT SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(
            f"  {r['a']} vs {r['b']}: "
            f"{r['a_wins']}W-{r['b_wins']}L-{r['draws']}D "
            f"({r['a_rate']:.0f}%)"
        )


if __name__ == "__main__":
    main()
