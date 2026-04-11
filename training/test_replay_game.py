"""Smoke test for training.replay_game (plan §7.8)."""

from __future__ import annotations

from types import SimpleNamespace

from training.config import AsyncConfig
from training import replay_game


class _FakeMove:
    def __init__(self, notation: str):
        self.notation = notation


class _FakeSearch:
    def __init__(self, moves: list[str]):
        self._moves = moves
        self._i = 0

    def set_rng_seed(self, seed: int) -> None:  # noqa: D401
        self.seed = seed

    def run_pcr(self, game, ply: int) -> dict:
        mv = _FakeMove(self._moves[self._i])
        self._i += 1
        return {
            "best_move": mv,
            "was_full_search": True,
            "value": 0.0,
            "nodes": 800,
            "policy_target": None,
            "top10": [(self._moves[self._i - 1], 800.0)],
        }


class _FakeGame:
    def __init__(self, num_plies: int):
        self._remaining = num_plies

    def is_game_over(self) -> bool:
        return self._remaining <= 0

    def apply(self, mv) -> None:
        self._remaining -= 1


def test_replay_completes_on_matching_trace():
    moves = ["f1-f2", "f11-f10", "g1-g2"]
    trace = {
        "game_id": 42,
        "model_version": 1,
        "rng_seed": 12345,
        "entries": [
            {"ply": 0, "selected_move": moves[0]},
            {"ply": 1, "selected_move": moves[1]},
            {"ply": 2, "selected_move": moves[2]},
        ],
    }

    verified = replay_game.replay(
        trace,
        model_path="/does/not/matter",
        search_factory=lambda: _FakeSearch(moves),
        game_factory=lambda: _FakeGame(num_plies=3),
    )
    assert verified == 3


def test_replay_default_search_factory_uses_trace_dirichlet(monkeypatch):
    moves = ["f1-f2"]
    created = []

    class _CtorSearch(_FakeSearch):
        def __init__(self, **kwargs):
            created.append(kwargs)
            super().__init__(moves)

    monkeypatch.setattr(
        replay_game,
        "hexchess",
        SimpleNamespace(
            MctsSearch=_CtorSearch,
            Game=lambda: _FakeGame(num_plies=1),
        ),
    )

    trace = {
        "game_id": 42,
        "model_version": 1,
        "rng_seed": 12345,
        "dirichlet_epsilon": 0.2,
        "dirichlet_alpha": 0.4,
        "entries": [{"ply": 0, "selected_move": moves[0]}],
    }

    verified = replay_game.replay(trace, model_path="/tmp/model.onnx")

    assert verified == 1
    assert created == [{
        "simulations": 800,
        "model_path": "/tmp/model.onnx",
        "dirichlet_epsilon": 0.2,
        "dirichlet_alpha": 0.4,
    }]


def test_replay_default_search_factory_falls_back_to_async_config(monkeypatch):
    moves = ["f1-f2"]
    created = []

    class _CtorSearch(_FakeSearch):
        def __init__(self, **kwargs):
            created.append(kwargs)
            super().__init__(moves)

    monkeypatch.setattr(
        replay_game,
        "hexchess",
        SimpleNamespace(
            MctsSearch=_CtorSearch,
            Game=lambda: _FakeGame(num_plies=1),
        ),
    )

    trace = {
        "game_id": 42,
        "model_version": 1,
        "rng_seed": 12345,
        "entries": [{"ply": 0, "selected_move": moves[0]}],
    }

    verified = replay_game.replay(trace, model_path="/tmp/model.onnx")
    cfg = AsyncConfig()

    assert verified == 1
    assert created == [{
        "simulations": 800,
        "model_path": "/tmp/model.onnx",
        "dirichlet_epsilon": cfg.dirichlet_epsilon,
        "dirichlet_alpha": cfg.dirichlet_alpha,
    }]
