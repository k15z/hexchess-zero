# Hex Chess AlphaZero Pipeline — Implementation Plan (v2 rebuild)

## 0. Guiding principles

1. **Treat AlphaZero as the skeleton, KataGo as the recipe.** Vanilla AZ works but is ~50× wasteful (notes/02, notes/08). Every choice below leans on KataGo / Lc0 unless our game characteristics (small board, mid branching, high decisive rate possible) actively contraindicate it.
2. **Debuggability is a first-class feature.** Every metric a future-us would want to inspect must be logged from day one — retrofitting logging during a confused regression is the dominant time sink in Lc0/KataGo history (notes/10, notes/11).
3. **Don't change hyperparameters mid-run.** Lock the config; if we want to test a change, start a fresh "run id" with a fresh anchor (notes/03 §5, notes/10 meta-principle).
4. **WDL everywhere, MLH from day one.** Adding heads later requires a fresh run (notes/03 §5, notes/05).

---

## 1. MCTS redesign

All in `engine/src/mcts.rs`. The current code already has arena nodes, batched eval, TT, virtual loss, and Dirichlet — we are adding KataGo features and re-tuning constants for hex chess.

### 1.1 PUCT formula

Use the **dynamic-`c_puct`** form from AlphaZero so that we benefit at high visit counts during eval/match play, but the constant term dominates training:

```
c_puct(s) = c1 + log((1 + N(s) + c2) / c2)
U(s,a)    = c_puct(s) · P(s,a) · sqrt(Σ_b N(s,b)) / (1 + N(s,a))
score(a)  = Q(s,a) + U(s,a)
```

with `c1 = 2.5`, `c2 = 19652`. Justification (notes/04 table, notes/03 §3):

- Branching factor 30–50 sits between Connect-4 (≈4) and chess (≈35). Lc0's *training-time* tree cpuct lives at 1.3–1.9, but Lc0 chess has stronger NN priors than we'll have early. KataGo used 1.1 for Go (250 branching). Oracle Connect-4 used ~4 (branching ~7).
- Linear interpolation by `log(branching)` puts us around **2.0–2.5**. We pick **2.5** to err toward exploration on a small under-trained network. Plan to sweep `{1.5, 2.0, 2.5, 3.0, 3.5}` after the first 1M positions and lock.
- Dynamic correction via `c2=19652` is a no-op below ~1k visits and ramps in cleanly past that — costs nothing, future-proofs us if we run higher-N evals (notes/04 §1).

Use `c_puct_root = 3.5` (Lc0 split: cpuct vs cpuct-at-root), and **at the root only** add the dynamic term scaled up. Justification: at the root we have Dirichlet noise + we want to explore; deeper nodes should be more exploitative.

### 1.2 FPU (First Play Urgency) reduction

Replace current "loss FPU" with **FPU reduction**:

- Training games: `Q_FPU(child) = Q_parent − 0.0` (i.e. = parent's Q). Rationale: Dirichlet does the exploration; FPU just provides a sane prior for unvisited siblings (notes/03 §4, notes/04 §FPU).
- Match/eval games: `Q_FPU = Q_parent − 0.2`.

### 1.3 Dirichlet noise

`α = 10 / branching ≈ 10 / 40 = 0.25` (notes/04 §Dirichlet, notes/12 §3). `ε = 0.25`.

Implement **shaped Dirichlet** (notes/02 §7): half the noise mass on `Dir(α/2)` (uniform-ish, broad blind-spot search), half concentrated on the top-K=10 prior moves, mixed at the root only. Disable noise on PCR fast moves and on all eval/match games.

### 1.4 Playout Cap Randomization (KataGo §3.1)

The single biggest data-efficiency win in KataGo's ablations (notes/02, notes/08).

```
With probability p_full = 0.25:
   N_sims = 800
   noise  = on
   record = (s, π_visit, …) → training buffer
With probability 1 − p_full:
   N_sims = 160
   noise  = off
   record = NOT a training sample for π or v_short
            (terminal z is still propagated to the *full-search* positions in this game)
```

Configurable via `MctsSearch::set_pcr(p_full, n_full, n_fast)`. Worker decides per move; only full-search positions go into the .npz batches with a `was_full_search=True` flag.

### 1.5 Forced playouts + policy target pruning (KataGo §3.2)

Forced visits per child:

```
n_forced(a) = ceil( k · P(s,a) · sqrt(N_total) ),  k = 2
```

A child cannot be skipped during selection until it has at least `n_forced(a)` visits (notes/02 §2).

**Policy target pruning:** when emitting `π` from visit counts, iterate forced children in increasing-visit order and subtract forced visits from each as long as that child does not become the new argmax. Renormalize. This removes the bias the forced exploration injects (notes/02 §2). Apply only to *full-search* training positions.

### 1.6 Virtual loss

Currently `VIRTUAL_LOSS = 3.0`. Reduce to `1.0` (KataGo default; notes/04 §VL). The current value 3 is high enough to harm search quality at the small batch sizes we use.

### 1.7 Transposition handling

Keep the current "TT for NN evaluations only, per-parent visit counts" approach (notes/04 §Trans). Already correct.

Improvements:
- Bound TT by entry count *and* by RSS — current 500k cap is fine but add an LRU eviction layer rather than full clear (full clear loses warm cache after every batch).
- Cache key: `(zobrist_hash, side_to_move, repetition_count)` — repetition state changes evaluation correctness.

### 1.8 Temperature schedule

Replace the current "high until move 60, then floor 0.35" with an Lc0-style smooth decay (notes/03 §6, notes/04 §temperature):

```
τ(ply) = τ_max · exp(−ply / τ_halflife) clamped to [τ_min, τ_max]
τ_max = 1.0,  τ_min = 0.1,  τ_halflife = 20 plies
```

Plus a hard "go greedy" cutoff at ply 60 to lock in late-game decisive play. Eval/match: `τ = 0` always, with LCB tiebreak.

### 1.9 LCB final move selection (notes/02 §6)

For evaluation/match play, pick the child maximizing
```
LCB(a) = Q(a) − 1.96 · sqrt(Var(a) / N(a))
```
where `Var(a)` is tracked online via Welford during backprop. Training games keep visit-proportional (or temperature-sampled) selection.

### 1.10 Resignation

Removed after rollout. The skipped calibration sample showed an unacceptable
false-positive rate, so resignation is no longer part of the worker or engine
design.

### 1.11 Terminal handling

- Threefold / 50-move / stalemate → exact value 0; no NN call (notes/04 §terminal).
- 2-fold within the current search tree → treated as a draw inside MCTS (notes/04, notes/10 §4) to prevent shuffle-loop sim waste.

### 1.12 Concrete defaults table

| Knob | Training | Eval/match |
|---|---|---|
| `c_puct`, `c_puct_root` | 2.5 / 3.5 | 2.5 / 3.5 |
| `c1, c2` (dynamic) | 2.5, 19652 | 2.5, 19652 |
| FPU reduction | 0.0 | 0.2 |
| Dirichlet `(α, ε)` | (0.25, 0.25) shaped | none |
| PCR `(p_full, n_full, n_fast)` | (0.25, 800, 160) | (1.0, 800, —) |
| Forced playout `k` | 2 | 0 |
| Policy target pruning | on | n/a |
| Virtual loss | 1 | 1 |
| Temperature | smooth decay (above) | 0 + LCB |

---

## 2. Network architecture

In `training/model.py`. **As built — see chunk 3 + the model-shrink TODO pass.**

### 2.1 Trunk

- **8 residual blocks × 144 filters** (~6.3M params total with all heads). Original plan called for 10×192 ~3M but the policy-head FC (`policy_ch · 121 · 4206`) dominates: at policy_ch=8 the FC alone is ~4M, blowing the trunk budget. The shrunk topology lands at 6.3M params.
- **SE in every block**, Leela-style scale+bias SE.
- **Global-pool bias branch in blocks 2 and 5** (was 3 and 6 for the 10-block layout, scaled to 1/4 and 5/8 depth for the 8-block layout).
- **BatchNorm** with the strict BN-eval-mode invariant from chunk 7. FixUp migration deferred indefinitely.

### 2.2 Heads

- **Policy head:** 1×1 conv (4 channels) → flatten (4·11·11=484) → FC → 4206. ~2M FC params.
- **Value WDL head:** 1×1 conv (32) → global avg pool → FC(256) → FC(3). **Mandatory** (notes/03, /05, /07, /12 §8).
- **Moves-left head (MLH):** 1×1 conv (16) → BN → ReLU → GAP → FC(64) → FC(1). Predicts plies-to-end. **Mandatory for hex chess** — see notes/12 §7 (engine shuffles to the move limit even at minimax-d3 without progress pressure).
- **Short-term value head (STV):** 1×1 conv (32) → BN → ReLU → GAP → FC(128) → FC(3) for WDL at horizon h=8 plies.
- **Auxiliary opponent-reply policy head:** 1×1 conv (**2 channels**, narrower than main) → flatten → FC → 4206. Target is the second-ply visit distribution exposed by `MctsSearch::aux_opponent_policy` (added in the TODO Rust pass).

Loss weights: λ_v=1.5, λ_p=1.0, λ_mlh=0.1, λ_stv=0.15, λ_aux_pol=0.15. WDL gets ε=0.05 label smoothing; nothing else does.

We're not adding KataGo's ownership/score (Go-specific, no analog). We're not adding Lc0's contempt/WDL-rescale yet — that's an eval-time-only feature (notes/03 §7).

### 2.3 Input planes

19 channels (`19 × 11 × 11`) is already implemented. Audit (each plane should have a unit test asserting it's nonzero on a known position):

| Plane | Content | Notes |
|---|---|---|
| 0–11 | piece type × color (one-hot, side-to-move's perspective) | flip board for black to play |
| 12 | side-to-move (constant) | redundant after flip but harmless |
| 13 | fullmove (normalized /200) | |
| 14 | halfmove clock (/100) | |
| 15 | en passant target | |
| 16 | repetition count (/3) | |
| 17 | validity mask | hex padding |
| 18 | in-check | cheap & known to help (notes/05 §inputs) |

Add (NEW), bringing channels to 22:
- 19: **last move from-square** (1-hot)
- 20: **last move to-square** (1-hot)
- 21: **plies-since-pawn-move** (/50) — gives the network MLH grounding from the input

These are all cheap (notes/05 §inputs principle).

### 2.4 Symmetries

**Correction from the original plan: the only valid symmetry is *central inversion* `(q,r) → (-q,-r)`, NOT horizontal reflection.** The naive horizontal-reflection transform `(q,r) → (-q-r, r)` does not preserve the pawn-promotion-pair set in Glinski. Central inversion does, because it maps white→black pawn directions and white→black promotion edges in one shot. See notes/12 §9 for the full reasoning and `engine/src/serialization.rs::test_mirror_symmetry_exhaustive` for the proof.

In CHW board encoding, central inversion is **flip both axis-2 (rows) and axis-3 (columns)** of the 11×11 grid. The Rust mirror table is precomputed at `MIRROR_INDEX` and exposed to Python via `hexchess.mirror_indices_array()`. The v2 trainer loader applies `mirror_batch` with p=0.5. Cheap 2× data multiplier.

No 6-fold rotational augmentation is possible — pawn direction breaks 5 of the 6 hex rotations.

---

## 3. Loss functions

In `training/trainer_loop.py` (move loss into `training/losses.py`).

```
L_total = λ_v · L_value
        + λ_p · L_policy
        + λ_mlh · L_mlh
        + λ_stv · L_stv
        + λ_aux_pol · L_aux_pol
        + λ_L2 · ||θ||²
```

Concrete weights (notes/07 §aux, KataGo paper §2 uses `c_g = 1.5`):

| Term | Weight | Loss |
|---|---|---|
| `L_value` (WDL) | 1.5 | cross-entropy against one-hot terminal `z` from STM perspective |
| `L_policy` | 1.0 | cross-entropy against PTP-pruned visit distribution `π`, illegal logits masked to `-1e9` (notes/10 §6, §13) |
| `L_mlh` | 0.1 | Huber on plies-to-end |
| `L_stv` | 0.15 | cross-entropy against horizon-8 WDL outcome (use Q at horizon-8 if not yet terminal) |
| `L_aux_pol` | 0.15 | cross-entropy against opponent-ply visit distribution |
| `L_L2` | 3e-5 | KataGo value (notes/07) |

Implementation notes:
- **No label smoothing** on policy (it suppresses sharp tactical signals). Yes label smoothing (`ε=0.05`) on the WDL head — empirically helps draw calibration and is what KataGo's later runs do for Go (notes/07 §2 — softmax doesn't saturate the same way as tanh, but smoothing still tames overconfidence).
- Use `log_softmax + nll_loss` to avoid FP underflow with ~4000 outputs (notes/10 §13).
- Mask illegals as `-1e9`, never `-inf` (notes/10 §6).
- Add `eps=1e-9` in any custom log.
- **Policy surprise weighting** (KataGo §7 follow-on, notes/02 §7): per-sample weight ∝ `KL(π_target || p_prior)` clipped to [0.5, 4]. Implement as a sample-weight column written by the worker (cost: one extra cross-entropy per recorded position, computed against the *current* worker net at recording time). Defer enabling until phase 3 — gives ~5–10% data efficiency, but adds complexity.

Healthy initial values to assert in CI smoke tests (notes/07 §healthy):
- Initial policy loss ≈ `log(num_legal) ≈ log(40) ≈ 3.7`. If first-batch policy loss is not in [3.0, 5.0], something is wrong with masking/encoding.
- Initial WDL loss ≈ `log(3) ≈ 1.099`. Same assertion.

---

## 4. Training loop

In `training/trainer_loop.py`.

### 4.1 Replay window — sublinear KataGo formula

Replace the current fixed `replay_buffer_size: 1_000_000` with KataGo's sublinear window (notes/06 §window):

```
N_window(N_total) = c · (1 + β · ((N_total / c)^α − 1) / α)
c = 25_000,  α = 0.75,  β = 0.4
```

Worked examples for our scale:
| `N_total` (positions ever generated) | window |
|---|---|
| 25k | 25k |
| 100k | ≈55k |
| 1M | ≈260k |
| 10M | ≈1.4M |
| 50M | ≈4.6M |

Justification: notes/12 §6 explicitly recommends `c = 25k` for our game scale (~10× smaller than KataGo Go). Sublinear avoids both stale-buffer (notes/10 §5) and washing-out (notes/10 §11).

The selection is by *recent positions* (parse timestamps from S3 filenames as the existing `select_recent_files` already does) — feed the window size from the formula at every reload.

### 4.2 Sample reuse

**`max_train_steps_per_new_data = 4`** (KataGo target, notes/06 §reuse). Token bucket as already implemented — keep `TrainBucket`, fix one bug: max_seed should not exceed window size.

### 4.3 Optimizer / LR / batch

| Setting | Value | Notes |
|---|---|---|
| Optimizer | SGD + momentum 0.9 | KataGo / AZ standard (notes/01) |
| Weight decay | 3e-5 | KataGo (notes/07) |
| Batch size | 256 | small game, faster wall-clock per cycle |
| LR | constant 1e-3, drop to 1e-4 after policy entropy plateaus or after a fixed step budget | (notes/06 §LR) |
| Gradient clip | global norm 5.0 | notes/10 §6 |
| Mixed precision | bf16 if cuda; fp32 on MPS | guarantee value/policy softmax in fp32 |
| Warmup | linear 0 → 1e-3 over first 2000 steps after bootstrap end | helps avoid early policy collapse (notes/10 §1) |

### 4.4 SWA (notes/06 §SWA)

Snapshot weights every 250k *training samples* (= every ~1000 batches at bs=256). Maintain a 4-snapshot rolling buffer. Promote the **EMA(decay=0.75) of the 4 most recent snapshots**, not the raw weights.

`promotion_weights = 0.4*w_t + 0.3*w_{t-1} + 0.2*w_{t-2} + 0.1*w_{t-3}` (after normalization with decay=0.75). BN running stats need to be re-estimated on a held-out batch after SWA averaging; if we move to FixUp this disappears.

### 4.5 Promotion frequency

Promote a new model **every 500k self-play positions** (notes/06 §promotion, notes/10 §15). Workers poll meta.json every 60s; the existing pull pattern is already correct — only change is to make promotion granularity match the replay window growth rate so the worker model never lags by more than ~one window slice.

### 4.6 Gating

**Off** in steady state (notes/06 §gating, notes/09 §gating). Continuous Elo service catches regressions. **On for the first 5 promotions after bootstrap**: candidate must score ≥ 50% (not 55%) over 200 SPRT-style games against current. Escape hatch: after 3 failed gates, promote anyway. This catches catastrophic bootstrap-to-selfplay handoff bugs.

### 4.7 Bootstrap

Imitation from minimax (notes/06 §bootstrap, notes/12 §10). Plan:

1. Worker pool generates `~1M` minimax positions (depth 5, with softmax-temperature exploration over the first 30 plies for diversity). Already implemented.
2. Trainer waits until ≥1M positions, then trains for **30k SGD steps** at lr=3e-3 against the imitation data.
3. Promote as v1.
4. Workers immediately switch to NN-guided MCTS self-play.

CI gate before declaring bootstrap done: v1 must beat material-only-depth-1 minimax in ≥80% of a 50-game match (this is a *very* low bar — failing it means encoding/training is broken).

### 4.8 Concrete cycle

```
[trainer] cycle = 1000 steps × bs 256 = 256k samples
[workers] generate ~64k new positions per cycle (4× target reuse)
[trainer] reload window every 1000 steps (existing reload_interval is fine)
[promote] every 2 cycles ≈ 500k samples
[snapshot for SWA] every cycle
```

---

## 5. Self-play workers

In `training/worker.py`.

### 5.1 Concurrency model

Per worker process: a single `MctsSearch` with `set_batch_size(32)`. ONNX Runtime is multi-threaded and saturates a CPU. On the kevz-gpu k3s GPU node, run the engine via the existing OnnxEvaluator with CUDA execution provider (already gated behind the `onnx` feature). On Mac Studio, MPS is unavailable for ORT; use CoreMLExecutionProvider if available else CPU.

**Concurrent games per worker:** 8 game loops in a single process, sharing one `MctsSearch` is not safe (search state is per-tree). Instead: 8 worker *threads* each owning their own `MctsSearch`, all calling `evaluate_batch` against a shared `Evaluator` that buffers and dispatches batches. This is the standard "leaf-batched" pattern. Implement a `BatchedEvaluator` wrapper around `OnnxEvaluator` with a 32-leaf coalescing queue and a 5ms max-wait timer. *(Substantial Rust work; deferred to phase 2 — phase 1 just runs N processes.)*

Phase 1: 1 game per process, N processes per node. 6 processes on the Mac Studio, 12 on the GPU node, 4 per Hetzner CPU node.

### 5.2 Model refresh

Already polls between batches. Reduce poll interval to **every game** (not every batch of 5) — at 90s/game and 500k-positions promotion, the worker is fine never lagging more than a few games.

### 5.3 Sample format on disk

Replace the current `(boards, policies, outcomes)` `.npz` with a **richer schema**, also `.npz`, columns:

| Column | shape | dtype | notes |
|---|---|---|---|
| `boards` | (N, 22, 11, 11) | int8 | input planes (int8 ok for binary planes; float for 4 normalized planes packed in last 4 channels) |
| `policy` | (N, num_move_indices) | float16 | PTP-pruned visit dist |
| `policy_aux_opp` | (N, num_move_indices) | float16 | opp-reply visit dist |
| `wdl_terminal` | (N, 3) | float32 | one-hot final outcome from STM |
| `wdl_short` | (N, 3) | float32 | horizon-8 outcome / Q |
| `mlh` | (N,) | int16 | plies-to-end |
| `was_full_search` | (N,) | bool | PCR flag — only True positions used as samples |
| `root_q` | (N,) | float16 | for diagnostics |
| `root_n` | (N,) | int32 | total visits at root |
| `root_entropy` | (N,) | float16 | of `π` |
| `nn_value_at_position` | (N,) | float16 | network's prior value (debugging) |
| `policy_kl_target_vs_prior` | (N,) | float16 | for surprise weighting |
| `legal_count` | (N,) | int16 | sanity check |
| `ply` | (N,) | int16 | move number in game |
| `game_id` | (N,) | uint64 | game-level joining key |

Per-game **metadata sidecar** `.json` (one per .npz):
```
{
  "game_id_range": [first, last],
  "model_version": v,
  "worker": "kevz-gpu-7",
  "started_at": iso,
  "duration_s": 92.3,
  "num_full_search_positions": n_full,
  "num_total_positions": n_total,
  "result": "white_win" | "draw" | "black_win",
  "termination": "checkmate" | "threefold" | "50move" | "stalemate" | "movelimit",
  "openings_hash": "abc123",
  "git_sha": "...",
  "rng_seed": 12345
}
```

Storage layout (extends current):
```
data/selfplay/v{N}/{ts}_{rand}_n{count}.npz
data/selfplay/v{N}/{ts}_{rand}_n{count}.meta.json
data/selfplay_traces/v{N}/{game_id}.json   # full per-move trace, see §7
```

Compression: `np.savez_compressed`. Estimate: ~40KB/position uncompressed → ~12KB compressed → 200 positions/file → 2.4MB/file, comfortable for S3.

### 5.4 PCR integration

Worker decides per move:
```
if rng() < 0.25:
   sims = 800; noise = on; record = True
else:
   sims = 160; noise = off; record = False
```
Passes the `record` flag through the search; samples accumulate only when `record=True`. Engine API addition: `MctsSearch::run_pcr(...) -> (best_move, maybe_full_result)`.

### 5.5 Determinism

Every worker process logs an `rng_seed` per game and writes it to the trace sidecar. A `replay_game.py` tool can re-run the exact game by injecting the seed and an immutable model version. This must work end-to-end — make it part of the phase-1 acceptance criteria (notes/11 reproducibility).

---

## 6. Evaluation

### 6.1 Anchors (notes/09 §anchors, notes/12 §11)

Maintain a fixed external anchor pool that **never changes for the run**:
- `anchor_random` — uniform random player.
- `anchor_minimax_d1_material` — material only.
- `anchor_minimax_d3_full` — material+mobility+king safety.
- `anchor_minimax_d5_full`.
- `anchor_minimax_d7_full`.
- `anchor_v0` — first promoted NN snapshot, frozen forever.

Anchors live in S3 as `anchors/{name}.json` config. The Elo service treats them as immortal players that get matched at high frequency early in a model's life and tapered off.

### 6.2 Continuous Elo service

Keep the existing `elo_service.py` design (immutable per-game records under `state/elo_games/`, projection rebuilt). Changes:

- **Pairing policy:** OpenSkill-uncertainty-driven matchmaking (already done) plus a hard rule: every new model plays at minimum 30 games against each anchor before its rating is "released" to the dashboard. Prevents premature regression alarms.
- **Use ≥800 sims** as already mandated by AGENTS.md.
- **SPRT** option for "is candidate stronger than current": API endpoint that sets up a pairing campaign with stop conditions `(elo0=0, elo1=10, α=β=0.05)`. Used during gating in phase-1 only.

### 6.3 Fixed benchmark suite

NEW: `benchmarks/positions.json` — a hand-curated set of ~50 positions covering:
- 10 opening positions (first ply for white from start; 9 unusual but legal first moves)
- 10 tactical puzzles (forced wins in 2–4 plies, hand-verified)
- 10 endgames (won, lost, drawn)
- 10 quiet middlegames
- 10 known-drawn positions

For every promoted version, the trainer (or a dedicated benchmark service) runs 800-sim MCTS on each position and records: top move, full visit dist, value WDL, MLH, root entropy. Stored at `benchmarks/results/v{N}.json`. The dashboard renders a heat map of "did this position's best move change between v(N-1) and v(N)?" and a per-position Elo-vs-version timeline.

### 6.4 Regression detection

Auto-alert (Slack via existing `slack.py`) when:
- Anchor Elo (vs `anchor_minimax_d5_full`) drops by >50 between consecutive versions.
- Anchor Elo (vs `anchor_v0`) drops at all between consecutive versions (it should monotonically rise).
- Draw rate against anchor pool jumps by >15 percentage points.
- Mean game length drops by >20% (often signals tactical regression).
- Any benchmark suite "best move" flips to a known-bad move.

---

## 7. Debuggability (CORE REQUIREMENT)

Adopting the principle from notes/11: log everything that's cheap, sample everything that's expensive, plot all of it.

### 7.1 Logging stack

- **Format:** structured JSON, one event per line, written to stdout (collected by k3s log forwarders) *and* to S3 under `logs/{service}/{hostname}/{date}/events.jsonl.gz` (rotated daily, gzipped on rotation).
- **Schema:** every event has `{ts, run_id, service, host, version, event, ...payload}`. `event` enum drives ingestion.
- **Library:** keep `loguru`, add a custom JSON sink. Add a `mlflow`-compatible local store under `logs/metrics/` for the dashboard to scrape.

### 7.2 Per training step (Tier 1, notes/11)

Logged every step (or aggregated every 50 steps to S3):

- `loss.policy`, `loss.value`, `loss.mlh`, `loss.stv`, `loss.aux_policy`, `loss.l2`, `loss.total`
- `policy.entropy_pred_mean`, `policy.entropy_target_mean`
- `policy.kl_pred_target` (training step KL)
- `policy.argmax_agreement_with_target` (network argmax vs MCTS argmax)
- `value.wdl_acc`, `value.brier`, `value.calibration_ece`
- `value.W_mean`, `value.D_mean`, `value.L_mean`
- `mlh.mae_plies`
- `optim.lr`, `optim.grad_norm_preclip`, `optim.grad_norm_postclip`, `optim.update_norm`
- `bn.running_mean_drift`, `bn.running_var_drift` (if BN)
- `data.window_size`, `data.cumulative_positions`, `data.bucket_tokens`
- `step_id`, `wall_ms`, `samples_per_s`

### 7.3 Per self-play game

- Everything in the `.meta.json` sidecar (§5.3).
- `mcts.mean_root_q`, `mcts.std_root_q`
- `mcts.mean_root_entropy`
- `mcts.tt_hit_rate` (worker-level)
- `mcts.mean_pv_length`
- `nn.mean_value`, `nn.mean_policy_entropy`
- `pcr.full_position_count`, `pcr.fast_position_count`
- `game.openings_hash` for opening-diversity tracking

### 7.4 Per MCTS search (Tier 2)

For every full-search position (the ones we actually train on), record the **per-move trace** in the trace sidecar (`data/selfplay_traces/...`):

```
{
  "game_id": ..., "ply": k, "side": "white",
  "fen": "...", "model_version": v,
  "nn_policy_top10": [(move, prob, child_q_after, child_n_after), ...],
  "mcts_visits_top10": [(move, n, q, lcb), ...],
  "policy_target_after_ptp": [(move, p), ...],
  "root_q": ..., "root_n": ...,
  "value_wdl": [W, D, L],
  "mlh_pred": ..., "stv_wdl": [...],
  "noise_used": bool, "fpu": ...,
  "search_ms": ...,
  "selected_move": ..., "selection_reason": "max_visits|temperature|lcb"
}
```

We don't store *all* MCTS internal nodes — too big. Top-10 children at the root is enough to debug 99% of pathologies.

### 7.5 Sanity-check invariants (auto-run)

A `training/health_checks.py` module runs after every reload and during every batch in CI mode. **Hard fail (CrashLoopBackoff)** the trainer if any of these breaks:

1. **Move encoding round-trip:** for 1000 random legal moves from random positions, `index_to_move(move_to_index(m)) == m`.
2. **Mirror table:** for 100 positions, the network policy on the mirrored board equals the mirrored policy of the original board (within tolerance) when fed through a freshly built deterministic model (offline test).
3. **Initial loss bounds:** first-batch policy loss ∈ [3.0, 5.0], WDL loss ∈ [0.95, 1.20]. Else encoding bug.
4. **No NaN / Inf** in any loss or gradient.
5. **No illegal-move probability mass:** sum of network softmax over illegal moves on a sample batch < 1e-3.
6. **Validity mask:** for every position, the input plane 17 contains exactly the 91 valid hex cells.
7. **Side-to-move flip:** mean of "STM piece planes" ≈ mean of "opponent piece planes" within 5% (no orientation bias).
8. **BN mode:** asserts `model.training == False` before any export call (notes/10 §7).
9. **Model output shape:** `(B, num_move_indices), (B, 3), (B, 1), (B, 3), (B, num_move_indices)`.
10. **TT cache hit rate** in worker is in [10%, 90%] (outside this range indicates a bug).
11. **Repetition detection:** play a forced-shuffle 8-ply sequence and assert `is_game_over() == True` due to threefold.

**Soft alerts (Slack)** when:

- Policy entropy < 0.5 *bits* across 1000 consecutive steps (collapse, notes/10 §1).
- Draw rate > 75% over 500 consecutive games (notes/10 §3).
- Game length 95th percentile = move limit (shuffle loops, notes/10 §4).
- Loss decreasing but anchor Elo not increasing for 10 promotions (notes/10 §5).
- White win % differs from black win % by >15 percentage points (orientation bug, notes/10 §9).
- Distinct-openings-per-100-games drops below 20 (notes/11 §hex specific).
- KL(old → new) per-promotion > 1.0 (overshooting, notes/11).

### 7.6 Dashboard panels

Extending `dashboard.py` / `dashboard.html`:

**Page 1 — Run health:**
1. Loss panel: policy / value / mlh / aux on dual axes, smoothed.
2. Policy entropy time series, with target entropy (`log(40)`) baseline.
3. Anchor Elo time series with ±1σ ribbons for every anchor. Frozen `anchor_v0` line should rise monotonically.
4. Self-play games / hour stacked by worker.
5. Mean game length (median + p5/p95 ribbon).
6. Draw / win-W / win-B fraction stacked area.

**Page 2 — Data:**
- Replay window size vs `N_total` cumulative.
- Train bucket tokens over time.
- Sample reuse (samples seen / samples generated).
- Per-version sample contribution to current window (stacked bar).
- Distribution of `was_full_search` (sanity: ≈ 25% of recorded).

**Page 3 — Model behavior:**
- Network value (WDL) calibration plot (binned predicted W vs empirical W).
- Policy entropy histogram by training step.
- NN argmax vs MCTS argmax agreement (notes/11 §NN vs MCTS).
- KL(old → new) per promotion.

**Page 4 — Search:**
- TT hit rate per worker.
- Mean root visits, mean PV length.
- LCB vs argmax-visit selection divergence rate.

**Page 5 — Benchmark suite:**
- Per-position move stability heatmap across versions.
- Per-position value stability.
- Per-version "tactical pass rate" (% of forced-win puzzles where the engine finds the win).

**Page 6 — Diff viewer:**
- Pick two model versions, run both on the benchmark suite and the per-game traces from a fixed RNG seed, render a side-by-side comparison with highlighted differences. This is the "audit two model versions on a fixed position suite" requirement.

**Page 7 — Replay buffer audit:**
- LIST `data/selfplay/v*` and show per-version file counts, byte size, position count.
- Browse a position by `game_id`, see its trace.
- Histogram of `policy_kl_target_vs_prior`, `legal_count`, `mlh`, `root_q`, etc.

### 7.7 Failure-mode auto-monitors

From `notes/10`, monitor (and alert) on every numbered failure mode:

| # | Detector |
|---|---|
| 1 Policy collapse | entropy < 0.5 for 1000 steps |
| 2 Value saturation | mean(max(WDL)) > 0.97 across batch |
| 3 Draw explosion | draw rate > 75% / 500 games |
| 4 Shuffle loops | p95 game length = move limit |
| 5 Stale buffer | loss ↓ but anchor Elo ↛ for 10 promotions |
| 6 NaN | hard-fail invariant |
| 7 BN eval | hard-fail invariant |
| 8 Encoding off-by-one | round-trip CI test |
| 9 Orientation | win-by-color delta > 15 pp |
| 10 Search exploration collapse | mean root entropy < 0.5 bit |
| 11 Recent overfit | gauntlet vs older versions degrades |
| 12 Cold-start stuck | bootstrap → v1 transition fails minimax-d1 gate |
| 13 Underflow | NaN detector + assertion that softmax stays in fp32 |
| 14 Repetition off-by-one | hard-fail invariant |
| 15 Trainer/selfplay drift | KL(old→new) > 1.0 per promotion |

### 7.8 Reproducibility / "replay one game"

Tool: `python -m training.replay --game-id <id>` reads the trace sidecar, downloads the exact model version it ran on, replays the game move-by-move with the recorded RNG seed, and asserts that resulting visit counts at each ply match within tolerance. If they don't match, it's a non-determinism bug — surface it loudly.

### 7.9 Replay buffer audit

Tool: `python -m training.audit_buffer` lists every file in the current window, prints summary stats per version (count, mean root_q, mean policy entropy, mean ply, draw rate). The dashboard wraps it as a panel.

---

## 8. Migration / rollout

### What to keep
- `engine/src/board.rs`, `movegen.rs`, `game.rs`, `serialization.rs` — game rules and move encoding are stable. **Do not touch the move-index bijection.** (notes/10 §8, notes/12 §2.)
- `engine/src/mcts.rs` arena structure, TT, batched eval scaffolding.
- `training/storage.py` S3 abstraction.
- `training/elo.py` and `elo_service.py` — design is good, just extend.
- `training/dashboard*` skeleton.
- `training/imitation.py` minimax bootstrap.

### What to delete / rewrite
- `training/model.py` — rewrite to add MLH, STV, aux-policy heads and 3 new input planes (from 19 → 22).
- `training/trainer_loop.py` — rewrite loss section, replace fixed buffer with sublinear window, add SWA, add health checks, add gating-for-first-5-versions.
- `training/worker.py` — rewrite to add PCR, new sample schema, per-game trace writer, per-game heartbeat.
- `engine/src/mcts.rs` — major surgery: PUCT constants, FPU reduction, PCR API, forced playouts, PTP, LCB, shaped Dirichlet, online variance tracking. The arena/TT plumbing stays.

### Order of implementation (each step is independently testable)

1. **Move-encoding freeze + CI guard.** Hash the move table; CI fails if the hash changes.
2. **Engine: PUCT + FPU + Dirichlet + temp + LCB + 2-fold-as-draw.** Unit tests for each, including a regression test against current self-play behavior on 5 fixed positions.
3. **Engine: PCR + forced playouts + PTP.** Unit test PTP correctness against a hand-computed example.
4. **Engine: shaped Dirichlet, online variance.**
5. **Network: rewrite `model.py` with new heads + 3 input planes.** Smoke test forward shape, parameter count, BN eval-mode test.
6. **Loss module: separate file, all 6 losses, label smoothing on WDL, mask logic, fp32-softmax guarantee.** Unit-test on synthetic data.
7. **Worker: new sample schema + trace sidecars + meta sidecars + PCR plumbing + game seeds.** Old `.npz` files become unreadable — that's fine, we're starting from scratch.
8. **Trainer: sublinear window, SWA, gating-for-first-5, full health-check module.**
9. **Dashboard: pages 1–4 (loss, data, model behavior, search).** Reuses existing infra.
10. **Eval: benchmark suite generator, per-version benchmark runner.**
11. **Eval: anchor Elo expansion, regression alerts.**
12. **Dashboard: pages 5–7 (benchmark, diff viewer, buffer audit).**
13. **Reproducibility: `replay_game.py`, determinism assertion.**
14. **Bootstrap dry-run:** generate 1M imitation positions on a single node, train v1, gate against minimax-d1, snapshot as `anchor_v0`.
15. **Phase-1 self-play kickoff** on the full cluster.

### "Ready to kick off a real run" checklist

- [ ] All 11 hard invariants pass on a fresh checkout in CI.
- [ ] `test_mcts_pcr_pruning` and `test_forced_playouts_count` pass.
- [ ] Mirror-symmetry test passes for the network on 100 random positions.
- [ ] First-batch loss bounds pass.
- [ ] `replay_game.py` round-trips a known game with byte-for-byte identical traces.
- [ ] Dashboard pages 1–4 render and ingest a 30-minute dry-run's data.
- [ ] Anchor pool exists in S3 with 6 anchors.
- [ ] Benchmark position file exists and `benchmarks/results/v0.json` was generated against the imitation-bootstrap model.
- [ ] All workers in the cluster (Mac Studio + Hetzner + kevz-gpu) heartbeat within 60s of `make worker`.
- [ ] Slack alert pipe confirmed by manually triggering one of the soft alerts.
- [ ] `run_id` set; no hyperparameters change after this point.

---

## 9. Risks and open questions

### Resolved by empirical measurement / implementation

1. ~~**Branching factor estimate.**~~ **Resolved.** Random play (n=75k positions) shows mean 39.8, median 40. `α = 10/40 = 0.25` confirmed. p5=6, p95=73 — wide distribution but the mean is what calibrates noise.
2. **Draw rate** — partially resolved. Random play shows ~11% formal draws + dominant timeout-by-shuffle. Real engine play likely 30%+ draws because Glinski's piece interaction is shuffle-prone. **MLH and STV are mandatory; WDL label smoothing kept at ε=0.05.**
3. ~~**Mirror symmetry.**~~ **Resolved.** Central inversion `(q,r) → (-q,-r)` is the only valid symmetry — exhaustively verified in `test_mirror_symmetry_exhaustive`. Horizontal reflection breaks promotion-pair closure. Updated notes/12 §9.
4. ~~**Move encoding hash guard.**~~ **Resolved.** `test_move_table_hash_is_stable` pins count=4206, hash=0x84d424aff5f7c2fd.

### Still open

5. **`c_puct` choice.** Locked at 2.5 / 3.5 by interpolation. Should sweep `{1.5, 2.0, 2.5, 3.0, 3.5}` against `anchor_v0` after the first 1M positions and re-lock if needed.
6. **Game length.** Notes/12 originally guessed 80–120 plies; empirical evidence (random and minimax+softmax) suggests games run much longer because both sides shuffle when balanced. Real number is unknown until MCTS self-play kicks off. Move-limit cap should be ≥ 250 plies, not 80.
7. **Aux-policy head cost.** Built at 2 channels (~1M FC params, narrower than the main 4-channel head). Worth it? Ablate at v3 if shrink is needed.
8. **BatchNorm vs FixUp.** Shipped with BN + strict eval-mode invariant. FixUp migration deferred; revisit only if we hit BN-related divergence.
9. **PCR record-keeping.** Currently only full-search positions become training samples. Fast positions could contribute to MLH/STV alone (denser horizon-8 supervision), but the worker doesn't do this. Defer until first run shows whether MLH signal is starved.
10. **Concurrent self-play within a process.** Deferred leaf-batched evaluator. Will revisit after measuring GPU saturation on kevz-gpu under 12 processes × 1 game.
11. **Replay window `c=25k`.** Locked. Empirical first-week tuning may bring it down to 15k if shorter Glinski games make 25k stale-prone.
12. **Bootstrap purity.** Shipped with imitation bootstrap (`training/imitation.py`). Pure-zero training is not pursued.
14. **Endgame / drawn benchmark fixtures.** Auto-generation produced 4 tactical (capture-finding) and 3 quiet middlegame positions. Endgame and drawn fixtures still need hand-curation (`benchmarks/positions.json` TODO).
15. **Dashboard pages 3-4.** Wired pages 1-2 against existing data; pages 3-4 wait on loss-snapshot and TT-stats pipelines that don't exist yet.
16. **Engine-level RNG determinism for replay.** Seeded RNG plumbed through Dirichlet noise (chunk 2 + Rust TODO pass). Full bit-identical replay would also need to seed any other RNG-consuming code path (none currently exist).

## 10. What was actually built (snapshot post-merge)

| Component | File(s) | Status |
|---|---|---|
| Move-table hash guard | `engine/src/serialization.rs` | ✓ Pinned at 4206 / `0x84d424aff5f7c2fd` |
| MCTS dynamic PUCT, FPU, shaped Dirichlet | `engine/src/mcts.rs` | ✓ |
| PCR + forced playouts + PTP + LCB + 2-fold-as-draw | `engine/src/mcts.rs` | ✓ |
| Seeded RNG (Dirichlet) | `engine/src/mcts.rs` | ✓ |
| Opponent-reply visit dist accessor | `engine/src/mcts.rs` + bindings | ✓ Used by worker |
| Mirror move-index table + exhaustive verification | `engine/src/serialization.rs` | ✓ Central inversion |
| 22-channel input encoder | `engine/src/serialization.rs` | ✓ |
| Network: 8×144 trunk, 5 heads | `training/model.py` | ✓ 6.3M params |
| Loss module (6 weighted terms + label smoothing) | `training/losses.py` | ✓ |
| Worker: PCR self-play, rich npz schema, sidecars | `training/worker.py` | ✓ |
| Trainer: sublinear window, SWA, gating-first-5 | `training/trainer_loop.py` | ✓ |
| Health checks (11 invariants, strict + runtime modes) | `training/health_checks.py` | ✓ |
| Structured JSONL logging | `training/logging_setup.py` | ✓ |
| Replay tool (CLI) | `training/replay_game.py` | ✓ |
| Buffer audit tool (CLI) | `training/audit_buffer.py` | ✓ |
| Benchmark suite (10 positions) | `benchmarks/`, `training/benchmark_runner.py` | ✓ (endgame TODO) |
| Dashboard pages 1-2 wired | `training/dashboard.html` | ✓ (3-4 placeholder) |
| Mirror augmentation in v2 loader | `training/data_v2.py` | ✓ |
| Config audit, run_id, validate() | `training/config.py` | ✓ |
| Trainer constants → cfg fields | `training/trainer_loop.py` | ✓ |
