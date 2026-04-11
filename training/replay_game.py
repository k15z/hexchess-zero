"""Replay one self-play game and assert determinism (plan §7.8).

Usage::

    uv run python -m training.replay_game --game-id <id>

Looks up the trace sidecar at ``data/selfplay_traces/v*/{game_id}.json``,
downloads the matching model version, and re-runs the game move-by-move
using the recorded RNG seed. At each recorded (full-search) ply, asserts
that the engine's best move matches the trace's ``selected_move``.

Known limitation (see chunk 5 notes on determinism): we do not byte-compare
visit counts; equality of the top move is the assertion. On mismatch we
print the trace top-10 and the engine top-10 side by side and exit non-zero.

Because only full-search positions are stored in the trace, fast-search
plies are replayed by simply trusting the engine's move there — those are
not validated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import storage
from .config import AsyncConfig
try:
    import hexchess  # type: ignore
except ImportError:  # pragma: no cover - binding optional during tests
    hexchess = None  # type: ignore


def find_trace_key(game_id: int) -> str:
    """Search ``data/selfplay_traces/v*/{game_id}.json`` for the game."""
    prefix = storage.SELFPLAY_TRACES_PREFIX
    target = f"{game_id}.json"
    for key in storage.ls(prefix):
        if key.endswith("/" + target):
            return key
    raise FileNotFoundError(f"no trace found for game_id={game_id} under {prefix}")


def load_trace(game_id: int) -> dict:
    key = find_trace_key(game_id)
    return json.loads(storage.get(key).decode())


def download_model(version: int, cfg: AsyncConfig) -> Path:
    cfg.ensure_cache_dirs()
    key = f"{storage.VERSIONS_PREFIX}{version}.onnx"
    local = cfg.model_cache_dir / f"v{version}.onnx"
    if not local.exists():
        storage.get_file(key, local)
    return local


def _move_str(mv) -> str:
    try:
        return mv.notation
    except Exception:
        return f"{mv.from_q},{mv.from_r}->{mv.to_q},{mv.to_r}"


def _format_top10(entries: list[tuple[str, float]]) -> str:
    lines = []
    for i, (name, score) in enumerate(entries[:10]):
        lines.append(f"  {i + 1:>2}. {name:<12} {score:.4f}")
    return "\n".join(lines) or "  (empty)"


def _trace_search_config(trace: dict) -> tuple[int, float, float]:
    """Return the search settings used for the recorded self-play game."""
    cfg = AsyncConfig()
    simulations = int(trace.get("num_simulations", cfg.num_simulations))
    epsilon = float(trace.get("dirichlet_epsilon", cfg.dirichlet_epsilon))
    alpha = float(trace.get("dirichlet_alpha", cfg.dirichlet_alpha))
    return simulations, epsilon, alpha


def replay(
    trace: dict,
    *,
    model_path: str | Path,
    search_factory=None,
    game_factory=None,
) -> int:
    """Replay the game; return the number of full-search plies verified.

    ``search_factory`` and ``game_factory`` exist so tests can inject fakes
    without requiring the native binding.
    """
    if search_factory is None:
        if hexchess is None:
            raise ImportError("hexchess bindings not available")
        search_factory = lambda: hexchess.MctsSearch(  # noqa: E731
        if hexchess is None:
            raise ImportError("hexchess bindings not available")
        game_factory = lambda: hexchess.Game()  # noqa: E731
        # — NOT eval_mode. New traces record the noise parameters explicitly;
        # older traces fall back to the current AsyncConfig defaults.
        simulations, dirichlet_epsilon, dirichlet_alpha = _trace_search_config(trace)
        search_factory = lambda: hexchess.MctsSearch(  # noqa: E731
            simulations=simulations,
            model_path=str(model_path),
            dirichlet_epsilon=dirichlet_epsilon,
            dirichlet_alpha=dirichlet_alpha,
        )
    if game_factory is None:
        if hexchess is None:
            raise ImportError("hexchess bindings not available")
        game_factory = lambda: hexchess.Game()  # noqa: E731

    search = search_factory()
    if hasattr(search, "set_rng_seed"):
        search.set_rng_seed(int(trace["rng_seed"]))

    entries_by_ply: dict[int, dict] = {
        int(e["ply"]): e for e in trace.get("entries", [])
    }
    max_ply = max(entries_by_ply) if entries_by_ply else -1

    game = game_factory()
    verified = 0
    ply = 0
    while not game.is_game_over() and ply <= max_ply + 1:
        outcome = search.run_pcr(game, ply)
        best = outcome["best_move"]
        best_str = _move_str(best)

        if ply in entries_by_ply:
            recorded = entries_by_ply[ply].get("selected_move")
            if recorded is not None and recorded != best_str:
                print(
                    f"MISMATCH at ply {ply}: trace={recorded} engine={best_str}",
                    file=sys.stderr,
                )
                trace_top = entries_by_ply[ply].get("mcts_visits_top10") or []
                engine_top = outcome.get("top10") or []
                print("\nTrace top-10:", file=sys.stderr)
                print(_format_top10(
                    [(m[0], float(m[1])) for m in trace_top]
                ), file=sys.stderr)
                print("\nEngine top-10:", file=sys.stderr)
                print(_format_top10(
                    [(m[0], float(m[1])) for m in engine_top]
                ), file=sys.stderr)
                sys.exit(2)
            verified += 1

        game.apply(best)
        ply += 1

    print(f"OK — replayed {ply} plies, verified {verified} full-search positions")
    return verified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", type=int, required=True)
    args = parser.parse_args(argv)

    cfg = AsyncConfig()
    trace = load_trace(args.game_id)
    model_version = int(trace["model_version"])
    model_path = download_model(model_version, cfg)
    replay(trace, model_path=model_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
