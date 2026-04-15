"""Run a local paired-color gate experiment between two model versions.

This script is intended as an offline exploration tool before changing the
production gate. It downloads immutable model snapshots from S3, runs paired
mini-matches locally, and reports two gate views as evidence accumulates:

1. The legacy score-threshold + confidence-interval rule
2. The current paired pentanomial-style GSPRT used by the evaluation service

Example:

    uv run python -m training.evaluate_gate --approved 1 --candidate 2
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .config import AsyncConfig
from .elo import MctsPlayer, hexchess, play_game
from .evaluation_service import _gate_decision, _gate_pentanomial_llr, _sprt_bounds

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


PAIR_BUCKET_ORDER = ("2.0", "1.5", "1.0", "0.5", "0.0")


@dataclass
class GateSnapshot:
    status: str
    total_games: int
    total_score: float
    llr: float
    lower_bound: float
    upper_bound: float


def _download_model(version: int, cfg: AsyncConfig) -> Path:
    from . import storage

    cfg.ensure_cache_dirs()
    key = f"{storage.VERSIONS_PREFIX}{version}.onnx"
    local = cfg.model_cache_dir / f"v{version}.onnx"
    if not local.exists():
        storage.get_file(key, local)
    return local


def _candidate_points(result: dict, *, candidate_is_white: bool) -> float:
    outcome = result["outcome"]
    if outcome == "draw":
        return 0.5
    if outcome == "white":
        return 1.0 if candidate_is_white else 0.0
    return 0.0 if candidate_is_white else 1.0


def _pair_bucket(pair_score: float) -> str:
    return f"{pair_score:.1f}"


def _move_str(mv) -> str:
    try:
        return mv.notation
    except Exception:
        return str(mv)
def _pair_sprt_snapshot(
    pair_counts: dict[str, int],
    total_games: int,
    *,
    alpha: float,
    beta: float,
    min_games: int,
    max_games: int,
    pass_score: float,
) -> GateSnapshot:
    lower, upper = _sprt_bounds(alpha, beta)
    llr = _gate_pentanomial_llr(pair_counts)
    pair_total = sum(pair_counts.values())
    pair_score_total = sum((float(bucket) / 2.0) * count for bucket, count in pair_counts.items())
    mean_score = (pair_score_total / pair_total) if pair_total else 0.0
    if total_games < min_games:
        status = "pending"
    elif llr >= upper:
        status = "approved"
    elif llr <= lower:
        status = "rejected"
    elif total_games >= max_games:
        status = "approved" if mean_score >= pass_score else "rejected"
    else:
        status = "pending"
    return GateSnapshot(
        status=status,
        total_games=total_games,
        total_score=pair_score_total * 2.0,
        llr=llr,
        lower_bound=lower,
        upper_bound=upper,
    )


def _current_gate_status(wins: int, losses: int, draws: int) -> GateSnapshot:
    status, total_games, total_score, half_width = _gate_decision(wins, losses, draws)
    label = status or "pending"
    return GateSnapshot(
        status=label,
        total_games=total_games,
        total_score=total_score * total_games,
        llr=0.0,
        lower_bound=total_score - half_width,
        upper_bound=total_score + half_width,
    )


def _format_pair_counts(pair_counts: dict[str, int]) -> str:
    return " ".join(f"{bucket}:{pair_counts.get(bucket, 0)}" for bucket in PAIR_BUCKET_ORDER)


def _format_score(total_score: float, total_games: int) -> str:
    if total_games <= 0:
        return "0.000"
    return f"{(total_score / total_games):.3f}"


def _play_logged_game(
    white,
    black,
    *,
    pair_index: int,
    game_index: int,
    max_moves: int = 600,
    log_moves: bool = True,
) -> dict:
    if hexchess is None:
        raise ImportError("hexchess bindings not available")

    game = hexchess.Game()
    move_count = 0
    white_time = 0.0
    black_time = 0.0
    white_moves = 0
    black_moves = 0

    print(f"  pair {pair_index:>3} game {game_index}: {white.name} (W) vs {black.name} (B)")

    while not game.is_game_over() and move_count < max_moves:
        is_white = game.side_to_move() == "white"
        player = white if is_white else black
        t0 = time.monotonic()
        mv = player.pick_move(game)
        dt = time.monotonic() - t0
        mv_str = _move_str(mv)
        move_count += 1

        if is_white:
            white_time += dt
            white_moves += 1
            side = "W"
        else:
            black_time += dt
            black_moves += 1
            side = "B"

        if log_moves:
            print(
                f"    move {move_count:>3} [{side}] {player.name:<8} "
                f"{mv_str:<14} {dt:>6.2f}s"
            )

        game.apply(mv)

    status = game.status()
    if status == "checkmate_white":
        outcome = "white"
    elif status == "checkmate_black":
        outcome = "black"
    else:
        outcome = "draw"

    print(
        f"  pair {pair_index:>3} game {game_index} result: {outcome} "
        f"after {move_count} moves (status={status})"
    )

    return {
        "outcome": outcome,
        "moves": move_count,
        "white_time": round(white_time, 2),
        "black_time": round(black_time, 2),
        "white_moves": white_moves,
        "black_moves": black_moves,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved", type=int, required=True, help="Approved/incumbent model version")
    parser.add_argument("--candidate", type=int, required=True, help="Candidate model version")
    parser.add_argument("--simulations", type=int, default=800, help="MCTS simulations per move")
    parser.add_argument("--max-pairs", type=int, default=50, help="Maximum paired mini-matches to run")
    parser.add_argument("--sprt-alpha", type=float, default=0.05, help="SPRT false-positive rate")
    parser.add_argument("--sprt-beta", type=float, default=0.05, help="SPRT false-negative rate")
    parser.add_argument(
        "--sprt-min-games",
        type=int,
        default=20,
        help="Minimum games before the pentanomial GSPRT may stop",
    )
    parser.add_argument(
        "--stop-when-decided",
        action="store_true",
        help="Stop as soon as both current gate and the paired pentanomial GSPRT are decided",
    )
    parser.add_argument(
        "--quiet-moves",
        action="store_true",
        help="Suppress per-move logging and only print game/pair summaries",
    )
    args = parser.parse_args(argv)

    if not (0.0 < args.sprt_alpha < 1.0 and 0.0 < args.sprt_beta < 1.0):
        raise SystemExit("--sprt-alpha and --sprt-beta must be in (0, 1)")

    cfg = AsyncConfig()
    approved_path = _download_model(args.approved, cfg)
    candidate_path = _download_model(args.candidate, cfg)

    approved = MctsPlayer(
        name=f"v{args.approved}",
        simulations=args.simulations,
        model_path=str(approved_path),
    )
    candidate = MctsPlayer(
        name=f"v{args.candidate}",
        simulations=args.simulations,
        model_path=str(candidate_path),
    )

    wins = 0
    draws = 0
    losses = 0
    total_score = 0.0
    pair_counts = {bucket: 0 for bucket in PAIR_BUCKET_ORDER}

    current_decision_pair: int | None = None
    current_decision_status: str | None = None
    sprt_decision_pair: int | None = None
    sprt_decision_status: str | None = None

    print(
        f"Gate experiment: candidate=v{args.candidate} vs approved=v{args.approved} "
        f"| sims={args.simulations} | max_pairs={args.max_pairs}"
    )
    print(
        "Paired pentanomial GSPRT: "
        f"alpha={args.sprt_alpha:.3f}, beta={args.sprt_beta:.3f}, "
        f"min_games={args.sprt_min_games}"
    )
    print("")

    for pair_index in range(1, args.max_pairs + 1):
        if args.quiet_moves:
            first = play_game(candidate, approved)
            second = play_game(approved, candidate)
        else:
            first = _play_logged_game(
                candidate,
                approved,
                pair_index=pair_index,
                game_index=1,
                log_moves=True,
            )
            second = _play_logged_game(
                approved,
                candidate,
                pair_index=pair_index,
                game_index=2,
                log_moves=True,
            )

        pair_game_points = [
            _candidate_points(first, candidate_is_white=True),
            _candidate_points(second, candidate_is_white=False),
        ]
        pair_score = sum(pair_game_points)
        total_score += pair_score
        pair_counts[_pair_bucket(pair_score)] += 1

        for pts in pair_game_points:
            if pts == 1.0:
                wins += 1
            elif pts == 0.5:
                draws += 1
            else:
                losses += 1

        total_games = wins + draws + losses
        current = _current_gate_status(wins, losses, draws)
        sprt = _pair_sprt_snapshot(
            pair_counts,
            total_games,
            alpha=args.sprt_alpha,
            beta=args.sprt_beta,
            min_games=args.sprt_min_games,
            max_games=100,
            pass_score=0.55,
        )

        if current.status != "pending" and current_decision_pair is None:
            current_decision_pair = pair_index
            current_decision_status = current.status
        if sprt.status != "pending" and sprt_decision_pair is None:
            sprt_decision_pair = pair_index
            sprt_decision_status = sprt.status

        print(
            f"pair {pair_index:>3}: "
            f"{_pair_bucket(pair_score):>3} pts "
            f"(g1={pair_game_points[0]:.1f}, g2={pair_game_points[1]:.1f}) | "
            f"W/D/L={wins}/{draws}/{losses} | "
            f"score={_format_score(total_score, total_games)}"
        )
        print(
            "          current="
            f"{current.status:<8} ci=[{current.lower_bound:.3f}, {current.upper_bound:.3f}]"
            f" | sprt={sprt.status:<8} llr={sprt.llr:.3f} "
            f"[{sprt.lower_bound:.3f}, {sprt.upper_bound:.3f}]"
        )
        print(f"          pair-buckets {_format_pair_counts(pair_counts)}")

        if (
            args.stop_when_decided
            and current_decision_pair is not None
            and sprt_decision_pair is not None
        ):
            break

    print("")
    print("Summary")
    print("-------")
    print(f"games: {wins + draws + losses} ({(wins + draws + losses) // 2} full pairs)")
    print(f"candidate W/D/L: {wins}/{draws}/{losses}")
    print(f"score: {total_score:.1f} / {wins + draws + losses} = {_format_score(total_score, wins + draws + losses)}")
    print(f"pair buckets: {_format_pair_counts(pair_counts)}")

    if current_decision_pair is None:
        print("current gate: no decision")
    else:
        print(
            f"current gate: {current_decision_status} at pair {current_decision_pair} "
            f"({current_decision_pair * 2} games)"
        )

    if sprt_decision_pair is None:
        print("paired pentanomial GSPRT: no decision")
    else:
        print(
            f"paired pentanomial GSPRT: {sprt_decision_status} at pair {sprt_decision_pair} "
            f"({sprt_decision_pair * 2} games)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
