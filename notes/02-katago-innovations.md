# 02 — KataGo's innovations

KataGo ([Wu 2019, "Accelerating Self-Play Learning in Go"](https://arxiv.org/abs/1902.10565))
reproduced ELF-OpenGo / AlphaZero strength in **~50x less compute** by stacking many small
improvements. The follow-up [`KataGoMethods.md`](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
documents further refinements through ~v1.13.

The paper's ablation runs (Table 2 in Wu 2019, measured at ~2.5B equivalent 20b×256c self-play
queries, ≈ 2 days) attribute a multiplicative speedup to each technique:

| Technique | Speedup |
|---|---|
| Playout cap randomization | **1.37x** |
| Forced playouts + policy target pruning | 1.25x |
| Global pooling | **1.60x** |
| Auxiliary policy targets | 1.30x |
| Ownership / score targets (Go-specific) | 1.65x |
| Game-specific input features | 1.55x |
| **Product of factors** | **~9.1x** (Wu calls this an underestimate) |

The "50x" in the abstract comes from compounding these with architectural and other gains
(SE blocks, larger nets, longer training).

## 1. Playout cap randomization (PCR)

**Problem:** With a fixed playout budget per move (say 800), every position contributes both a
value sample and a policy target. But the *value* target benefits from playing more games (more
diverse positions), while the *policy* target benefits from deeper search per move. These are in
tension.

**Solution:** Randomize the playout cap per move.

- With probability `p ≈ 0.25`: do a "full" search of `N` playouts (e.g. 600 → 1000), and **save
  this position for training** (both policy + value targets).
- With probability `1 − p`: do a "fast" search of `n` playouts (e.g. 100 → 200), use only for
  *move selection*, and **do NOT add to the training buffer for the policy target** (the value
  target from terminal outcome is still recorded for positions that *were* used).

KataGo main run: `(N, n) = (600, 100)` initially, annealed to `(1000, 200)` after the first two
days of training, with `p = 0.25` ([Wu 2019 §3.1](https://arxiv.org/abs/1902.10565)). Fast
searches **disable Dirichlet noise and other explorative settings** to maximise playing strength
on the cheap moves. Only full-search turns are recorded for training.

**Why it works:** ~75% of moves play 6x faster, dramatically increasing game throughput, while
the 25% "full search" positions still provide high-quality policy targets. Both heads benefit.

## 2. Forced playouts + policy target pruning

**Forced playouts:** At the root, every child that has any prior `P(s,a) > 0` is *forced* to
receive at least

```
n_forced(a) = ⌈ k · P(s,a) · sqrt(N_total) ⌉
```

playouts (`k ≈ 2`). This guarantees rare-but-promising moves get explored even when PUCT would
otherwise starve them. It is essentially exploration injected in proportion to prior belief.

**Policy target pruning:** When constructing the policy target `π` from visit counts, *subtract*
the forced visits from each child if doing so doesn't change which move was best, then renormalize.
This prevents the network from learning to over-weight the forced exploration.

**Why both are needed:** Forced playouts alone bias the policy target toward unimportant moves.
Pruning fixes the bias. Together: 1.25x speedup.

## 3. Global pooling layers

**Problem:** A 3x3 conv has receptive field 3. Even with 20 residual blocks, information
propagates only ~40 cells, and 19x19 Go has critical global features (overall territory, life
status of distant groups). Pure-conv nets are bad at this.

**Solution:** In selected residual blocks (KataGo: blocks at depths 1/3 and 2/3 through the
trunk), include a "global pooling bias" branch:

1. Take the conv output of shape `(C, H, W)`.
2. Compute `mean` and `max` across `H,W` for each channel → `(2C,)` vector.
3. Pass through a small FC layer → per-channel bias.
4. Add this bias to the spatial features.

This is conceptually similar to **Squeeze-and-Excitation** ([Hu et al. 2018](https://arxiv.org/abs/1709.01507))
and KataGo later adopted SE blocks throughout. The single-paper number is **1.6x speedup**.

The lesson: **whatever your game, if it has global state (whose turn it is, material balance,
phase of game), give the network a way to mix global features at every block.**

## 4. Auxiliary targets

KataGo predicts more than just `(p, v)`:

- **Ownership** (Go): per-cell prediction of who will own the territory at game end. Acts as
  dense supervision — every move provides 361 extra labels.
- **Score** (Go): exact final score margin distribution.
- **Auxiliary policy target**: the policy after the *opponent's* next move (helps the network
  learn forcing sequences).
- **Short-term value targets** (added later): exponential averages of the value at horizons of
  ~6, 16, 50 plies. Provide lower-variance feedback than terminal-only `z`.

**Lesson for non-Go games:** Look hard for auxiliary signals that come for free from your game
state. Examples for chess: piece-square ownership, material delta, "will this pawn promote?",
"will the king move within N plies?". The cost is a few extra heads; the benefit is dense
supervision that regularizes the trunk.

## 5. Squeeze-and-Excitation blocks

By KataGo v1.4 every residual block was an SE block. SE inserts after the second conv:

```
y = conv2(relu(conv1(x)))
s = sigmoid(FC2(relu(FC1(global_avg_pool(y)))))
out = x + (y * s)
```

Per-channel scalar gating that depends on global context. Cheap (`FC` is small, ratio 1:8 or 1:4)
and consistently helps in board-game nets ([Cazenave 2021](https://arxiv.org/abs/2102.03467)).

## 6. LCB (Lower Confidence Bound) move selection

For the *final* move choice (not search expansion), KataGo uses an LCB criterion: pick the child
that maximizes a lower-confidence-bound on its true value, not just the highest visit count. This
prevents picking a high-visit move that turned out badly in the search.

```
LCB(a) = Q(a) − c · sqrt(Var(a) / N(a))
```

This matters most in time-limited match play where a high-visit but bad move could otherwise
slip through.

## 7. Other follow-on techniques (`KataGoMethods.md`)

From the methods doc, in roughly chronological order of addition:

- **Shaped Dirichlet noise** — half uniform, half concentrated on top-policy moves, instead of
  fully uniform. Better blind-spot exploration.
- **Root policy softmax temperature** — KataGoMethods.md describes 1.25 early, decaying
  exponentially to 1.1 with a halflife equal to board dimensions. (The original Wu 2019 main run
  used a flat root softmax temperature of 1.03; the 1.25→1.1 schedule is a later refinement.)
- **Policy surprise weighting** — sample weights ∝ KL(target || prior) so the network sees more
  of the surprising positions.
- **Subtree value bias correction** — record per-pattern bias of NN value vs MCTS value, subtract
  at inference. ~30–60 Elo.
- **Variance-scaled cPUCT** — scale `c_puct` by empirical utility variance at the node. ~75 Elo
  with uncertainty weighting in v1.9.
- **Uncertainty-weighted playouts** — NN predicts its own short-term squared error; high-error
  playouts get downweighted.
- **Auxiliary soft policy target** — predict `policy^(1/T)` with `T=4`, weight 8x the normal
  policy target. Helps the model discriminate among low-probability moves.
- **Optimistic policy** — uses short-term value predictions to bias exploration toward
  promising-but-uncertain moves.
- **Nested bottleneck residual blocks** — four 3x3 convs with nested skip connections. More
  compute-efficient than standard ResNet for board games.

## Headline numbers (Wu 2019, §1 and Appendix C)

- Total compute: ~27 V100-GPUs average for 19 days ≈ 1.4 GPU-years. The run started with 16
  V100s for self-play + gating/training overhead and grew to 24 self-play GPUs after the first
  two days.
- Total samples: 241M training samples across 4.2M self-play games (final b20×c256 stage).
- Progressive net growth: (6,96) → (10,128) → (15,192) → (20,256) at ~0.75, 1.75, 7.5 days.
- Final network size (20 blocks × 256 channels) matches AlphaZero/ELF.
- "~50x less compute" vs ELF OpenGo in the abstract; ELF used ~2000 V100s for 13–14 days
  (~74 GPU-years).

For a small game with branching factor ~30–50, you should expect to need much less compute, and
the *relative* speedups from these techniques should be at least as good (the search-vs-network
tension PCR addresses is even worse on small branching factors).
