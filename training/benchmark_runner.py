"""Benchmark suite runner (plan §6.3).

Loads ``benchmarks/positions.json``, replays ``moves`` from the initial
position, runs MCTS on each position, records top move / top-10 / value
WDL / MLH / root_entropy, and writes ``benchmarks/results/v{N}.json``
locally and to S3 under the same relative key.

Positions with ``"skip": true`` (placeholders) are skipped with a TODO
log line.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import storage
from .config import AsyncConfig
try:
    import hexchess  # type: ignore
except ImportError:  # pragma: no cover
    hexchess = None  # type: ignore


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_positions(path: Path | None = None) -> list[dict]:
    """Load the benchmark positions JSON. Returns the list of entries."""
    path = path or _project_root() / "benchmarks" / "positions.json"
    data = json.loads(path.read_text())
    positions = data.get("positions", [])
    return positions


def validate_positions(positions: list[dict]) -> None:
    required = {"id", "category", "description", "moves"}
    for p in positions:
        missing = required - set(p)
        if missing:
            raise ValueError(f"position {p.get('id', '?')} missing fields: {missing}")
        if not isinstance(p["moves"], list):
            raise ValueError(f"position {p['id']}: 'moves' must be a list")
        if p["category"] not in {"opening", "tactical", "endgame", "middlegame", "drawn"}:
            raise ValueError(f"position {p['id']}: unknown category {p['category']}")


def _entropy(probs: list[float]) -> float:
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    return h


def _apply_moves(game, moves: list[str]) -> None:
    """Apply a sequence of moves (by notation) to a Game."""
    for notation in moves:
        legal = game.legal_moves()
        match = None
        for mv in legal:
            if getattr(mv, "notation", None) == notation:
                match = mv
                break
        if match is None:
            raise ValueError(f"move {notation!r} not legal from current position")
        game.apply(match)


def run_position(position: dict, model_path: str, simulations: int = 800) -> dict:
    """Run MCTS on one position and return a result dict."""
    if hexchess is None:
        raise ImportError("hexchess bindings not available")
    game = hexchess.Game()
    search = hexchess.MctsSearch(
    _apply_moves(game, position.get("moves", []))

    search = hexchess.MctsSearch(
        simulations=simulations,
        model_path=model_path,
        eval_mode=True,
    )
    outcome = search.run_pcr(game, 0)
    best = outcome["best_move"]
    best_str = getattr(best, "notation", str(best))
    top10 = [list(x) for x in (outcome.get("top10") or [])][:10]
    return {
        "id": position["id"],
        "category": position["category"],
        "best_move": best_str,
        "top10": top10,
        "value": float(outcome.get("value", 0.0)),
        "nodes": int(outcome.get("nodes", 0)),
        "wdl": outcome.get("wdl"),
        "mlh": outcome.get("mlh"),
        "root_entropy": outcome.get("root_entropy"),
    }


def run_all(model_path: str, simulations: int = 800) -> list[dict]:
    positions = load_positions()
    validate_positions(positions)
    results: list[dict] = []
    for p in positions:
        if p.get("skip"):
            print(f"[skip] {p['id']}: {p['description']}")
            continue
        try:
            results.append(run_position(p, model_path, simulations))
        except Exception as exc:  # pragma: no cover
            print(f"[fail] {p['id']}: {exc}")
    return results


def save_results(results: list[dict], version: int) -> Path:
    root = _project_root() / "benchmarks" / "results"
    root.mkdir(parents=True, exist_ok=True)
    rel = f"benchmarks/results/v{version}.json"
    local = root / f"v{version}.json"
    payload = {"version": version, "results": results}
    local.write_text(json.dumps(payload, indent=2))
    try:
        storage.put(rel, local.read_bytes())
    except Exception as exc:  # pragma: no cover
        print(f"[warn] failed to upload {rel}: {exc}")
    return local


def _current_model_version(cfg: AsyncConfig) -> tuple[int, str]:
    meta = storage.get_json(storage.LATEST_META)
    version = int(meta.get("version", 0))
    cfg.ensure_cache_dirs()
    local = cfg.model_cache_dir / f"v{version}.onnx"
    storage.get_file(storage.LATEST_ONNX, local)
    return version, str(local)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override local ONNX path (skips S3 download).")
    parser.add_argument("--version", type=int, default=None,
                        help="Version label for output file; defaults to latest.")
    args = parser.parse_args(argv)

    cfg = AsyncConfig()
    if args.model_path:
        model_path = args.model_path
        version = args.version or 0
    else:
        version, model_path = _current_model_version(cfg)
    results = run_all(model_path, simulations=args.simulations)
    out = save_results(results, version=version)
    print(f"Wrote {out} ({len(results)} positions)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
