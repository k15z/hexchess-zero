"""Shared rating types, game play, and OpenSkill rating computation."""

from __future__ import annotations

import time
from typing import Protocol

from openskill.models import PlackettLuce

try:
    import hexchess
except ImportError:
    hexchess = None


# ---------------------------------------------------------------------------
# Rating model (module-level singleton)
# ---------------------------------------------------------------------------

_model = PlackettLuce()

# Conservative rating: mu - Z * sigma
CONSERVATIVE_Z = 2.0

# Sigma below this means OpenSkill's posterior on the player is tight enough
# that we treat their mu as meaningful. Default rating starts at sigma≈8.33
# and drops below this after ~20-30 games vs a well-calibrated opponent.
EVALUATED_SIGMA_THRESHOLD = 2.5

EVAL_MARKER = "[eval]"
PROV_MARKER = "[prov]"


def conservative_rating(mu: float, sigma: float) -> float:
    """Raw conservative rating: mu - Z*sigma (OpenSkill scale, no rescaling)."""
    return mu - CONSERVATIVE_Z * sigma


def is_evaluated(sigma: float) -> bool:
    """True if the posterior is tight enough that mu is meaningful."""
    return sigma <= EVALUATED_SIGMA_THRESHOLD


def eval_marker(sigma: float) -> str:
    """Return the [eval] or [prov] tag for display alongside a rating."""
    return EVAL_MARKER if is_evaluated(sigma) else PROV_MARKER


def conservative_ratings(ratings: dict[str, dict]) -> dict[str, float]:
    """Project {name: {mu, sigma}} to {name: mu - 2σ} for scalar comparison."""
    return {n: conservative_rating(r["mu"], r["sigma"]) for n, r in ratings.items()}


def rank_by_conservative(
    ratings: dict[str, dict],
) -> list[tuple[str, dict]]:
    """Return ratings sorted by conservative rating (mu - 2σ), best first."""
    return sorted(
        ratings.items(),
        key=lambda kv: conservative_rating(kv[1]["mu"], kv[1]["sigma"]),
        reverse=True,
    )


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
        return hexchess.minimax_search(game, self.depth).best_move


class MctsPlayer:
    def __init__(self, name: str, simulations: int, model_path: str | None = None):
        self.name = name
        self.search = hexchess.MctsSearch(
            simulations=simulations,
            model_path=model_path,
            eval_mode=True,
        )

    def pick_move(self, game):
        return self.search.run(game, temperature=0.0).best_move


def baselines(simulations: int = 800) -> list[Player]:
    return [
        MinimaxPlayer(name="Minimax-2", depth=2),
        MinimaxPlayer(name="Minimax-3", depth=3),
        MinimaxPlayer(name="Minimax-4", depth=4),
        MctsPlayer(name="Heuristic", simulations=simulations),
    ]


# ---------------------------------------------------------------------------
# Match play
# ---------------------------------------------------------------------------


def play_game(white: Player, black: Player, max_moves: int = 600,
              random_opening_plies: int = 0) -> dict:
    """Play one game. Returns dict with outcome and per-player timing stats.

    If random_opening_plies > 0, plays that many random moves before handing
    off to the players. Randomizes +/-1 ply so both sides get an equal chance
    of making the first non-random move.
    """
    import random

    game = hexchess.Game()

    if random_opening_plies > 0:
        n_random = random_opening_plies + random.randint(-1, 0)
        for _ in range(n_random):
            if game.is_game_over():
                break
            moves = game.legal_moves()
            mv = random.choice(moves)
            game.apply(mv)

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
        game.apply(mv)
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
        "white_time": round(white_time, 2),
        "black_time": round(black_time, 2),
        "white_moves": white_moves,
        "black_moves": black_moves,
    }


# ---------------------------------------------------------------------------
# OpenSkill rating helpers
# ---------------------------------------------------------------------------


def new_rating() -> dict:
    """Create a fresh rating as a serializable dict."""
    r = _model.rating()
    return {"mu": r.mu, "sigma": r.sigma}


def predict_draw(a: dict, b: dict) -> float:
    """OpenSkill match-quality metric: high when both sigmas are large and mus are close."""
    ra = _model.rating(mu=a["mu"], sigma=a["sigma"])
    rb = _model.rating(mu=b["mu"], sigma=b["sigma"])
    return _model.predict_draw(teams=[[ra], [rb]])


def update_ratings(
    a_rating: dict, b_rating: dict, outcome: str,
) -> tuple[dict, dict]:
    """Update ratings for a 1v1 result. Returns new (a_rating, b_rating).

    outcome: "a_wins", "b_wins", or "draw"
    """
    ra = _model.rating(mu=a_rating["mu"], sigma=a_rating["sigma"])
    rb = _model.rating(mu=b_rating["mu"], sigma=b_rating["sigma"])

    if outcome == "a_wins":
        ranks = [1, 2]
    elif outcome == "b_wins":
        ranks = [2, 1]
    else:
        ranks = [1, 1]

    [[new_a], [new_b]] = _model.rate(teams=[[ra], [rb]], ranks=ranks)
    return (
        {"mu": new_a.mu, "sigma": new_a.sigma},
        {"mu": new_b.mu, "sigma": new_b.sigma},
    )


def replay_results(
    ratings: dict[str, dict],
    pair_results: dict[str, dict],
) -> None:
    """Replay pairwise results to build ratings. Mutates ratings in place.

    pair_results: {pair_key: {"a_wins": int, "b_wins": int, "draws": int, ...}}
    where pair_key is "a:b" (sorted names).
    """
    for pair_key, result in pair_results.items():
        a, b = pair_key.split(":")
        if a not in ratings or b not in ratings:
            continue
        for _ in range(result.get("a_wins", 0)):
            ratings[a], ratings[b] = update_ratings(ratings[a], ratings[b], "a_wins")
        for _ in range(result.get("b_wins", 0)):
            ratings[a], ratings[b] = update_ratings(ratings[a], ratings[b], "b_wins")
        for _ in range(result.get("draws", 0)):
            ratings[a], ratings[b] = update_ratings(ratings[a], ratings[b], "draw")


def compute_elo(
    players: list[str],
    results: list[dict],
) -> dict[str, dict]:
    """Compute ratings from batch pairwise results. Returns {name: {mu, sigma}}.

    This is a convenience wrapper for scripts that run round-robin tournaments.
    It replays all results through OpenSkill and returns the raw OpenSkill
    posteriors. Callers that want a single scalar for ordering should call
    ``conservative_rating(mu, sigma)``.

    results: [{"a": str, "b": str, "a_wins": int, "b_wins": int, "draws": int}, ...]
    """
    ratings = {p: new_rating() for p in players}

    # Convert list-of-dicts to pair_results format (keyed by sorted names)
    pair_results = {}
    for r in results:
        a, b = r["a"], r["b"]
        key = ":".join(sorted([a, b]))
        # If sorting swapped the names, swap a_wins/b_wins to match
        if sorted([a, b]) == [a, b]:
            pair_results[key] = r
        else:
            pair_results[key] = {
                "a_wins": r["b_wins"], "b_wins": r["a_wins"], "draws": r["draws"],
            }

    replay_results(ratings, pair_results)

    return ratings


def format_elo_table(ratings: dict[str, dict]) -> str:
    """Format OpenSkill ratings as a ranked table string.

    ratings: {name: {"mu": float, "sigma": float}}

    Players are sorted by conservative rating (mu - 2σ). Each row is tagged
    [eval] or [prov] based on whether σ ≤ EVALUATED_SIGMA_THRESHOLD; prefer
    [eval] players when comparing strengths — a [prov] player's μ has not
    converged yet.
    """
    lines = []
    for rank, (name, r) in enumerate(rank_by_conservative(ratings), 1):
        cr = conservative_rating(r["mu"], r["sigma"])
        lines.append(
            f"  {rank}. {name:<20s} {cr:>6.2f}  "
            f"(μ={r['mu']:5.2f} ±{r['sigma']:4.2f}) {eval_marker(r['sigma'])}"
        )
    return "\n".join(lines)
