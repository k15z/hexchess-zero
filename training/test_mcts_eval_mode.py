"""Regression tests for the Python MctsSearch `eval_mode` switch.

These guard against a past bug where the Python binding hard-coded
`SearchConfig::training()` and only conditionally layered Dirichlet noise
on top, so callers that passed `dirichlet_epsilon=0.0` (Elo, benchmarks,
replay) silently ran in training mode — with Dirichlet at
root, forced playouts, policy-target pruning, no LCB, etc.

If `config_summary()` is missing from the binding, the binding is older
than this fix and the test fails loudly rather than silently skipping.
"""

from __future__ import annotations

import pytest

hexchess = pytest.importorskip("hexchess")


def _make(**kwargs):
    # simulations=1 keeps construction cheap; we never call run().
    return hexchess.MctsSearch(simulations=1, **kwargs)


def test_training_mode_defaults_match_search_config_training():
    search = _make()
    cfg = search.config_summary()
    # Training mode uses engine's SearchConfig::training() defaults.
    assert cfg["c_puct"] == pytest.approx(2.5)
    assert cfg["c_puct_root"] == pytest.approx(3.5)
    # Training defaults from engine/src/mcts.rs SearchConfig::training().
    assert cfg["forced_playout_k"] == pytest.approx(2.0)
    assert cfg["policy_target_pruning"] is True
    assert cfg["use_lcb"] is False
    assert cfg["fpu_reduction"] == pytest.approx(0.0)
    # Training has Dirichlet ON by default.
    assert cfg["dirichlet"] is not None
    assert cfg["dirichlet"]["epsilon"] == pytest.approx(0.25)


def test_eval_mode_matches_search_config_eval():
    search = _make(eval_mode=True)
    cfg = search.config_summary()
    # Eval defaults from engine/src/mcts.rs SearchConfig::eval().
    assert cfg["c_puct"] == pytest.approx(2.5)
    assert cfg["c_puct_root"] == pytest.approx(3.5)
    assert cfg["forced_playout_k"] == pytest.approx(0.0)
    assert cfg["policy_target_pruning"] is False
    assert cfg["use_lcb"] is True
    assert cfg["fpu_reduction"] == pytest.approx(0.2)
    # Eval has no Dirichlet at root.
    assert cfg["dirichlet"] is None


def test_eval_mode_with_dirichlet_epsilon_zero_stays_clean():
    # This is the regression: old code left training-mode Dirichlet in
    # place when epsilon=0.0. eval_mode must win.
    search = _make(eval_mode=True, dirichlet_epsilon=0.0)
    cfg = search.config_summary()
    assert cfg["dirichlet"] is None
    assert cfg["forced_playout_k"] == pytest.approx(0.0)


def test_eval_mode_allows_explicit_dirichlet_override():
    # If a caller passes a positive epsilon with eval_mode, they opt in
    # to Dirichlet on top of the eval config.
    search = _make(eval_mode=True, dirichlet_epsilon=0.1, dirichlet_alpha=0.4)
    cfg = search.config_summary()
    assert cfg["dirichlet"] is not None
    assert cfg["dirichlet"]["epsilon"] == pytest.approx(0.1)
    assert cfg["dirichlet"]["alpha"] == pytest.approx(0.4)
    # But the rest of eval config stays put.
    assert cfg["forced_playout_k"] == pytest.approx(0.0)
    assert cfg["use_lcb"] is True


def test_c_puct_and_batch_overrides_respected_in_both_modes():
    for eval_mode in (False, True):
        search = _make(eval_mode=eval_mode, c_puct=2.0, batch_size=16)
        cfg = search.config_summary()
        assert cfg["c_puct"] == pytest.approx(2.0)
        # set_c_puct also bumps the root constant by +1.
        assert cfg["c_puct_root"] == pytest.approx(3.0)
        assert cfg["batch_size"] == 16
