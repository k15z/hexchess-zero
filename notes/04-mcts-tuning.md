# 04 — MCTS tuning

## PUCT formula variants

The vanilla AlphaZero PUCT:

```
U(s,a) = c_puct · P(s,a) · sqrt(Σ_b N(s,b)) / (1 + N(s,a))
score(a) = Q(s,a) + U(s,a)
```

The dynamic-`c_puct` variant from the AlphaZero appendix:

```
c_puct(s) = log((1 + N(s) + c2) / c2) + c1     where (c1, c2) = (1.25, 19652)
```

In practice this matters only at very high visit counts (>10k). For 800–1600 sims, just use a
constant `c_puct ∈ [1.5, 4.0]`.

## Choosing `c_puct`

Empirically observed defaults:

| System | `c_puct` | Notes |
|---|---|---|
| AlphaZero paper | dynamic, `c1=1.25, c2=19652` | rarely matters at low N |
| Lc0 UCI default | 3.0 (`CPuctBase=19652`, `CPuctFactor=2.0`) | [lc0-options](https://lczero.org/dev/wiki/lc0-options/) |
| Lc0 training runs | tree `cpuct ≈ 1.3–1.9`, `cpuctatroot ≈ 1.9–2.5` | internal values, T60+ |
| KataGo main run (Wu 2019) | fixed `cPUCT = 1.1` | per paper §2 |
| KataGo recent | dynamic variance-scaled cPUCT | added in ~v1.9, ~75 Elo combined with uncertainty weighting |
| Oracle AZ-Connect4 | ~4.0 | branching ~7 |
| OpenSpiel default | 2.5 | |

Heuristics:
- **Higher branching factor → higher `c_puct`** (more children competing for visits).
- **Higher visit budget → higher `c_puct`** (you can afford more exploration).
- **Sharp tactical games (chess) → lower `c_puct`** (deep lines matter more than broad).

The cleanest approach: pick a midpoint, generate ~200 games each at `c_puct ∈ {1.5, 2.0, 2.5, 3.0,
3.5, 4.0}` against a fixed opponent, take the best.

## Dirichlet noise

Added at the **root only** to encourage exploration of moves the network would otherwise never
play:

```
P'(s_root, a) = (1 − ε) · P(s_root, a) + ε · η_a    where η ~ Dir(α)
```

AlphaZero used `ε = 0.25` everywhere.

The `α` parameter scales with branching factor. The (now-canonical) heuristic:

```
α ≈ 10 / avg_legal_moves
```

| Game | avg moves | α used |
|---|---|---|
| Go 19x19 | ~250 | 0.03 |
| Shogi | ~80 | 0.15 |
| Chess | ~35 | 0.30 |
| Connect Four | ~7 | ~1.0–1.75 |
| Glinski hex chess | ~30–50 | ~0.2–0.3 |

KataGo's "shaped Dirichlet": split the noise so half is uniform (`α / 2`) and half is concentrated
on the top-prior moves. Helps the noise actually find blind spots rather than wasting exploration
on obvious junk moves.

**Disable Dirichlet for fast/cheap moves under PCR**, and disable it entirely at evaluation time.

## FPU reduction

For an unvisited child during selection, you need *some* `Q` to plug into `Q + U`. Options:

1. **Loss FPU** (`Q = −1`): never explore unless prior is very high. AlphaZero default.
2. **Parent value FPU** (`Q = Q_parent`): maximally optimistic, over-explores.
3. **FPU reduction** (`Q = Q_parent − r`): a tunable middle. `r ≈ 0.2–0.5` for play, `r=0` for
   training games (let Dirichlet do the exploration).

Lc0 and KataGo both use FPU reduction. It's substantially better than the AZ original.

## Virtual loss

For batched / parallel MCTS, you can't have multiple workers traversing the same path. Apply
**virtual loss** when entering a node: pretend the node lost some games (`N += vl`, `Q decreases`)
so other workers pick a different branch. Undo on backprop.

Typical `vl = 1` to `3`. Higher = more diversity but worse search quality. KataGo defaults to
`vl = 1` plus collision-handling logic.

## Temperature schedule for move selection

After search, sample the move from `N(s,a)^(1/τ) / Σ N(s,b)^(1/τ)`:

- **τ = 1**: sample proportional to visits (high diversity, weak play).
- **τ → 0**: argmax over visits (greedy, strong play, no exploration).

AlphaZero: `τ=1` for first 30 moves of self-play, then `τ→0`.

Lc0 (training runs, T60+): start `τ` around 0.9–1.3, decay smoothly toward ~0.4–0.65 over the
first ~20–60 moves. Exact schedule varies by run (see Lc0 project history). The principle —
smooth decay rather than a cliff — is the durable lesson.

For evaluation/match play, always use `τ=0` (with LCB or just argmax visits).

## Transposition handling

Two options for hashing equivalent positions:

1. **Tree, no transpositions**: simplest. Same position via different paths is two nodes. Wastes
   compute but is bug-free. AlphaZero original.
2. **DAG with transposition table**: shared evaluation cache, but be careful — you can NOT just
   share visit counts (that breaks PUCT's parent-N requirement).

Most modern systems compromise: **transposition table for NN evaluations only**, separate per-
parent visit counts. KataGo and Lc0 both do this. Saves a huge amount of NN compute on
chess/Go where positions repeat.

For Glinski hex chess, NN cache hit rate will be lower than chess (smaller endgames, fewer
repetitions) but still probably worth implementing.

## Drawn / terminal handling

- **Terminal nodes**: assign exact value `{−1, 0, +1}`, never query the NN.
- **Threefold repetition**: in training games, treat as draw immediately (don't waste sims).
- **N-fold repetition during search**: tricky. Most engines treat 2-fold within a search subtree
  as a draw (this is "search draws as draw") to avoid wasting search on repetition cycles.

## Resignation

Removed from this project.

We tried the standard KataGo/AlphaZero-style calibration idea, but the
observed false-positive rate in skipped calibration games was far too high
for self-play labels to remain trustworthy. Games now always play to natural
termination and throughput is recovered elsewhere.
