# 01 — AlphaZero, end-to-end

## The core loop

AlphaZero ([Silver et al. 2018](https://www.science.org/doi/10.1126/science.aar6404)) couples
three components in a tight feedback loop:

1. **Neural network** `f_θ(s) → (p, v)` predicts a policy distribution `p` over legal moves and a
   scalar value `v ∈ [−1, 1]` (estimated outcome from the side-to-move's perspective).
2. **MCTS search** uses `p` and `v` as priors and leaf evaluations to produce an *improved* policy
   `π` (the visit-count distribution at the root) and an *improved* value (the search-backed Q at
   the root).
3. **Self-play games** are generated using MCTS for move selection. The terminal game outcome `z`
   is recorded for every position visited.

Training samples are tuples `(s, π, z)`. The loss is:

```
L = (z − v)² − π · log p + c · ||θ||²
```

i.e. value MSE + policy cross-entropy + L2 weight decay. AlphaZero used `c = 1e-4`.

## What was novel (versus AlphaGo / AlphaGo Zero / TD-Gammon / etc.)

- **No human data, no rollouts, no handcrafted features.** AlphaGo Zero
  ([Silver et al. 2017](https://www.nature.com/articles/nature24270)) eliminated supervised
  pre-training; AlphaZero generalized to chess and shogi in a single architecture.
- **Single network with two heads.** Earlier AlphaGo had separate policy and value networks; the
  unified ResNet was both simpler and stronger.
- **MCTS as policy improvement operator.** The network doesn't try to imitate Stockfish or any
  oracle — it imitates the *output of search using itself as a guide*. Each iteration, search
  produces moves slightly better than the network's raw policy; training distills that
  improvement back. This is the policy iteration view: MCTS = policy improvement, gradient step =
  policy evaluation.
- **No rollouts at leaves.** Earlier MCTS engines (Crazy Stone, Pachi, AlphaGo Lee) used random
  Monte Carlo rollouts to estimate leaf values. AlphaZero replaced these entirely with `v` from the
  network — counter-intuitive at the time, but it works because the value head is far better
  calibrated than random play.

## Key hyperparameters in the original paper

From the AlphaZero Science paper supplementary materials:

| Parameter | Value | Notes |
|---|---|---|
| Residual blocks | 19 (chess/shogi), 39 (Go) | 256 filters |
| MCTS sims/move (training) | 800 | Per move; ~0.4s on a TPU |
| `c_puct` | 1.25 (with log term) | Dynamic via the `(c1, c2)=(1.25, 19652)` formula |
| Dirichlet `α` | 0.3 (chess), 0.15 (shogi), 0.03 (Go) | Heuristic: `α ≈ 10 / avg_legal_moves` |
| Dirichlet `ε` | 0.25 | Mixing weight at root |
| Temperature | τ=1 for first 30 plies, then τ→0 | Selection from visit counts `N^(1/τ)` |
| Batch size | 4096 | trained on 64 second-gen TPUs |
| LR schedule | 0.2 → 0.02 → 0.002 → 0.0002 | Stepwise (exact boundaries not given in the paper; schedule inherited from AlphaGo Zero) |
| Optimizer | SGD + momentum 0.9 | Weight decay from AGZ (`c ≈ 1e-4`) |
| Total training | 700k mini-batch steps; chess 9h, shogi 12h, Go 34h | 5000 first-gen TPUs for self-play, 64 second-gen TPUs for training (Table S3) |

The single hyperparameter changed across games was the Dirichlet `α`, scaled to branching
factor. See `04-mcts-tuning.md` for the rule of thumb.

## The PUCT formula

AlphaZero's variant of PUCT:

```
U(s,a) = c_puct · P(s,a) · sqrt(ΣN(s,b)) / (1 + N(s,a))
a* = argmax_a [Q(s,a) + U(s,a)]
```

with the dynamic `c_puct` (introduced in AlphaZero paper, appendix):

```
c_puct(s) = log((1 + N(s) + c2) / c2) + c1
```

with `c1=1.25, c2=19652` (inherited from AlphaGo Zero's methods; AlphaZero says parameters are
"identical to AlphaGo Zero"). In practice most reproductions use a constant `c_puct ≈ 1.5–4`
and the dynamic term barely matters until you're at huge visit counts.

## What AlphaZero did *not* do (and what's been added since)

Things people learned to add after the original paper:

- **No auxiliary targets** — KataGo added ownership/score; Lc0 added moves-left-head.
- **No global pooling / SE** — both major efficiency wins (KataGo).
- **No playout cap randomization** — biggest single training-efficiency win (KataGo).
- **No SWA / weight averaging** — standard now.
- **Scalar value head, not WDL** — Lc0 showed WDL is strictly better when draws exist.
- **Constant fixed cost per move** — KataGo's PCR is much better.
- **Gating with 55% threshold** — modern systems just promote continuously.

In short: the original AZ recipe is *correct* but *very* compute-inefficient by 2024 standards.
Treat it as the skeleton; layer KataGo's tricks on top.
