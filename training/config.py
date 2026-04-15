"""Hyperparameters and paths for the training pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_run_id() -> str:
    """Run identifier for logging/metrics grouping (notes/13 §7.1, §8)."""
    return os.environ.get("RUN_ID", "dev")


def _project_root() -> Path:
    """Return the project root (parent of training/)."""
    return Path(__file__).resolve().parent.parent


def _cache_root() -> Path:
    """Local cache for downloaded S3 objects (models, .npz files)."""
    return _project_root() / ".cache"


@dataclass
class _BaseConfig:
    """Shared hyperparameters for training and model architecture."""

    # --- Run identity ---
    run_id: str = field(default_factory=_default_run_id)

    # --- MCTS ---
    num_simulations: int = (
        800  # quality-over-speed default: deeper search for cleaner self-play targets
    )

    # --- Self-play ---
    temperature_threshold: int = (
        60  # after this many moves, temperature drops to temperature_low
    )
    temperature_high: float = 1.0
    temperature_low: float = (
        0.35  # Lc0-style floor — 0.01 produced one-hot targets in 65-70% of positions
    )
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # --- Training ---
    batch_size: int = 256
    # Observed: at 1e-3 the policy oscillates 3.1–3.5 across trainer summaries without
    # converging. Halving to 5e-4 for smoother optimization on the large
    # (~4200-output) policy FC. KataGo's per-sample LR was 6e-5 ≈ 1.5e-2
    # at batch 256; our 5e-4 is more conservative but safer for a model
    # this early in training.
    learning_rate: float = 5e-4
    momentum: float = 0.9
    l2_regularization: float = 3e-5  # KataGo weight decay (plan §4.3)
    grad_clip_norm: float = 5.0  # plan §4.3
    lr_warmup_steps: int = 2_000  # plan §4.3
    # Produce candidates materially less often. The previous 300k cadence
    # allowed very weak nets to replace self-play before the replay window had
    # enough fresh signal to stabilize, so we now wait for a meaningfully
    # larger tranche of new data between promotions.
    promote_every_new_positions: int = 2_500_000
    runtime_health_check_every_steps: int = 500

    # --- Imitation mix (bootstrap → decay → off) ---
    # Fraction of training batches sourced from minimax imitation data
    # vs self-play. Anchors the policy to the teacher signal during early
    # training when self-play is noisy — observed: without this, v2 was
    # WORSE than heuristic. But holding the mix permanently at a fixed
    # value pins the policy to the minimax teacher long after self-play
    # has overtaken it; notes/13 calls for a "bootstrap then switch"
    # schedule. We linearly decay from ``imitation_mix_start`` at v1 to
    # ``imitation_mix_end`` by ``imitation_mix_decay_end_version``, after
    # which the trainer is pure self-play.
    imitation_mix_start: float = 0.3
    imitation_mix_end: float = 0.0
    imitation_mix_decay_end_version: int = 8

    # --- Replay window (sublinear KataGo formula, plan §4.1) ---
    window_c: int = 25_000
    window_alpha: float = 0.75
    window_beta: float = 0.4

    # --- SWA (plan §4.4) ---
    swa_snapshot_every_samples: int = 250_000
    swa_buffer_size: int = 4
    swa_ema_decay: float = 0.75
    # Rolling number of recent training batches kept for re-estimating BN
    # running stats after SWA averaging. update_bn_stats uses cumulative
    # averaging (momentum=None), so a single batch produces very noisy
    # running mean/var and stale running stats silently degrade every
    # promoted model. 32 batches * batch_size=256 = ~8k samples, which is
    # enough for a stable BN reset on a 192-filter ResNet.
    swa_bn_refresh_batches: int = 32

    # --- Playout Cap Randomization (plan §1.4/§5.4) ---
    # Plan targets 800/160 for production. Observed on kevz-infra at kickoff:
    # 800-sim self-play yields ~64 pos/min across 4 CPU workers, making the
    # early promotion cadence multi-day. Dropping to 200/50 in early training gives ~4x
    # throughput; the learned policy manifests fine at 200 sims for a near-
    # random-init v1. Ratchet back to 800 once the NN is strong enough that
    # sim depth limits performance.
    pcr_p_full: float = 0.5
    # Production target: 800 sims. Earlier reductions (200, 400) produced
    # noisy policy targets — v6 self-play is 66% draws (43% insufficient
    # material) because the search can't find winning tactics at low depth.
    # At 800 sims the NN's learned prior compounds over the search tree,
    # producing sharper visit distributions that the network can learn from.
    # Throughput drops to ~90 pos/min with 22 workers — v7 promotion cadence ~2 days.
    pcr_n_full: int = 800
    pcr_n_fast: int = 160

    # --- Network architecture ---
    # Sized to ~5M params: small enough to iterate fast on the cluster + Mac
    # Studio, big enough to learn the ~4200-move policy without bottlenecking.
    num_residual_blocks: int = 8
    num_filters: int = 144
    se_channels: int = 32
    global_pool_channels: int = 32
    global_pool_blocks: tuple[int, ...] = (2, 5)
    policy_channels: int = 4
    aux_policy_channels: int = 2  # narrower opponent-reply head
    value_channels: int = 32
    board_channels: int = 22
    board_height: int = 11
    board_width: int = 11


@dataclass
class AsyncConfig(_BaseConfig):
    """Configuration for the async distributed training pipeline.

    All shared state lives in S3 (DigitalOcean Spaces / Cloudflare R2 / etc).
    Local .cache/ directory holds downloaded files for ONNX Runtime and np.load.
    """

    # --- Async-specific ---
    worker_batch_size: int = (
        2  # games per flush (small so bootstrap data lands quickly)
    )
    summary_interval_steps: int = 3000  # steps between trainer summaries; not a promotion gate
    reload_interval: int = 1000  # reload replay data and re-check promotion eligibility every N steps
    max_train_steps_per_new_data: float = (
        4.0  # target passes per data point (KataGo-style bucket)
    )
    min_positions_to_start: int = (
        2_000_000  # use the full preserved imitation corpus for a stronger v1 bootstrap
    )

    # --- Imitation bootstrap ---
    # Depth 5 is prohibitively slow on Glinski (>10 min per game observed).
    # Depth 3 produces still-reasonable targets (captures, obvious tactics)
    # and is ~30x faster, which lets us get through bootstrap in hours
    # instead of days.
    imitation_depth: int = 3
    imitation_exploration_plies: int = 30  # plies using softmax sampling for diversity
    imitation_temperature: float = (
        200.0  # softmax temperature for policy targets and exploration sampling
    )
    imitation_wdl_lambda: float = 0.5  # blend: λ*sigmoid(eval) + (1-λ)*game_outcome
    bootstrap_steps: int = (
        150_000  # more optimization passes now that bootstrap uses ~2M imitation positions
    )
    bootstrap_learning_rate: float = (
        0.003  # higher LR for clean supervised signal (3x self-play LR)
    )

    # --- Local cache ---

    @property
    def cache_dir(self) -> Path:
        return _cache_root()

    @property
    def model_cache_dir(self) -> Path:
        return _cache_root() / "models"

    @property
    def data_cache_dir(self) -> Path:
        return _cache_root() / "data"

    def ensure_cache_dirs(self) -> None:
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def steps_per_cycle(self) -> int:
        """Backward-compatible alias for the old cadence name."""
        return self.summary_interval_steps

    @steps_per_cycle.setter
    def steps_per_cycle(self, value: int) -> None:
        self.summary_interval_steps = value

    def imitation_mix_for_version(self, version: int) -> float:
        """Return the imitation-mix fraction for the given model version.

        Linearly interpolates from ``imitation_mix_start`` at v1 down to
        ``imitation_mix_end`` by ``imitation_mix_decay_end_version``; held
        constant outside that interval so callers can pass any version
        without clamping at the call site.
        """
        start = self.imitation_mix_start
        end = self.imitation_mix_end
        end_v = self.imitation_mix_decay_end_version
        if end_v <= 1 or version >= end_v:
            return end
        if version <= 1:
            return start
        t = (version - 1) / (end_v - 1)
        return start + (end - start) * t

    def validate(self) -> None:
        """Sanity-check config values; raise ``ValueError`` on bad input."""
        errors: list[str] = []

        def _check(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        _check(self.batch_size > 0, "batch_size must be > 0")
        _check(self.learning_rate > 0, "learning_rate must be > 0")
        _check(self.l2_regularization >= 0, "l2_regularization must be >= 0")
        _check(self.grad_clip_norm > 0, "grad_clip_norm must be > 0")
        _check(self.lr_warmup_steps >= 0, "lr_warmup_steps must be >= 0")
        _check(
            self.runtime_health_check_every_steps > 0,
            "runtime_health_check_every_steps must be > 0",
        )
        _check(
            self.promote_every_new_positions > 0,
            "promote_every_new_positions must be > 0",
        )
        _check(self.num_simulations > 0, "num_simulations must be > 0")

        _check(self.window_c > 0, "window_c must be > 0")
        _check(self.window_alpha > 0, "window_alpha must be > 0")
        _check(0 < self.window_beta, "window_beta must be > 0")

        _check(0 < self.pcr_p_full <= 1.0, "pcr_p_full must be in (0, 1]")
        _check(self.pcr_n_full > 0, "pcr_n_full must be > 0")
        _check(self.pcr_n_fast > 0, "pcr_n_fast must be > 0")
        _check(self.pcr_n_fast <= self.pcr_n_full, "pcr_n_fast must be <= pcr_n_full")

        _check(self.swa_buffer_size >= 1, "swa_buffer_size must be >= 1")
        _check(0.0 < self.swa_ema_decay <= 1.0, "swa_ema_decay must be in (0, 1]")
        _check(
            self.swa_snapshot_every_samples > 0,
            "swa_snapshot_every_samples must be > 0",
        )
        _check(self.swa_bn_refresh_batches >= 1, "swa_bn_refresh_batches must be >= 1")

        _check(
            self.max_train_steps_per_new_data > 0,
            "max_train_steps_per_new_data must be > 0",
        )
        _check(
            self.summary_interval_steps > 0,
            "summary_interval_steps must be > 0",
        )
        _check(
            0.0 <= self.imitation_mix_start <= 1.0,
            "imitation_mix_start must be in [0, 1]",
        )
        _check(
            0.0 <= self.imitation_mix_end <= 1.0, "imitation_mix_end must be in [0, 1]"
        )
        _check(
            self.imitation_mix_start >= self.imitation_mix_end,
            "imitation_mix_start must be >= imitation_mix_end "
            "(schedule is a decay, not a ramp)",
        )
        _check(
            self.imitation_mix_decay_end_version >= 1,
            "imitation_mix_decay_end_version must be >= 1",
        )
        _check(bool(self.run_id), "run_id must be non-empty")

        if errors:
            raise ValueError("Invalid AsyncConfig: " + "; ".join(errors))


if __name__ == "__main__":
    cfg = AsyncConfig()
    print("Training pipeline configuration:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
