# AlphaZero-style Self-Play Training: Research Notes

A runbook for building an AlphaZero-style self-play training pipeline for a board game,
distilled from the original AlphaGo Zero / AlphaZero papers, the KataGo paper and follow-on
methods doc, and years of Leela Chess Zero (Lc0) training experience.

These notes are intentionally **general** — they apply to any two-player perfect-information
game. Specific adaptation hints for small-board hex variants are in `12-hex-chess-adaptation.md`.

## Reading order

For a first pass, read in order:

1. [`01-alphazero-overview.md`](01-alphazero-overview.md) — How AlphaZero works end-to-end and why it was novel.
2. [`02-katago-innovations.md`](02-katago-innovations.md) — KataGo's ~50x efficiency improvements: PCR, forced playouts, policy target pruning, global pooling, auxiliary targets.
3. [`03-lc0-lessons.md`](03-lc0-lessons.md) — Lc0's multi-year journey: WDL head, MLH, FPU, cpuct, contempt, what worked and what didn't.
4. [`04-mcts-tuning.md`](04-mcts-tuning.md) — PUCT, Dirichlet noise, FPU reduction, temperature schedules, transpositions.
5. [`05-network-architecture.md`](05-network-architecture.md) — Residual blocks, SE, global pooling, value-head design.
6. [`06-training-loop.md`](06-training-loop.md) — Replay window sizing, sample reuse, LR schedules, SWA, gating.
7. [`07-loss-functions.md`](07-loss-functions.md) — Policy/value loss, WDL cross-entropy, auxiliary losses, healthy curves.
8. [`08-data-efficiency.md`](08-data-efficiency.md) — How fast can you surpass strong handcrafted baselines?
9. [`09-evaluation-and-elo.md`](09-evaluation-and-elo.md) — Measuring progress without fooling yourself.
10. [`10-common-failure-modes.md`](10-common-failure-modes.md) — Bug runbook with symptoms and fixes.
11. [`11-debugging-signals.md`](11-debugging-signals.md) — What to log, what healthy looks like.
12. [`12-hex-chess-adaptation.md`](12-hex-chess-adaptation.md) — Adapting AlphaZero to a small-board variant.

## Top-level lessons in one page

- **MCTS visit counts are the policy target, not argmax of NN policy.** That's the whole reason AlphaZero works: search distills into the network, network amplifies search.
- **Use WDL, not scalar value.** Three-headed Win/Draw/Loss cross-entropy is strictly better than MSE on a [-1,1] scalar for any game with draws ([Lc0 WDL post](https://lczero.org/blog/2020/04/wdl-head/)).
- **Playout cap randomization** is the single biggest free win after the original AlphaZero recipe — 1.37x speedup from one trick ([KataGo §5](https://arxiv.org/abs/1902.10565)).
- **Global pooling layers** are the second biggest — 1.6x speedup. Standard CNNs cannot share global state through 3x3 convs alone.
- **Window the replay buffer carefully.** Too small → overfit to recent self-play. Too large → stale data. KataGo uses a sublinear growing window: `c(1 + β((N/c)^α − 1)/α)` with `α=0.75, β=0.4, c=250k`.
- **Don't gate.** Modern systems (KataGo, recent Lc0) skip the AlphaGo Zero "55% win-rate gating" step. Continuous promotion is faster and just as stable.
- **Always have an external anchor.** A handcrafted baseline (random, material-only, minimax-at-fixed-depth) prevents Elo inflation in a closed self-play pool.
- **Log policy entropy, value MAE, draw rate, game length, and KL(old||new).** These four numbers diagnose ~80% of bugs. See [`11-debugging-signals.md`](11-debugging-signals.md).

## Source legend

Inline citations point to:

- AlphaGo Zero (Silver et al., Nature 2017): [paper](https://www.nature.com/articles/nature24270)
- AlphaZero (Silver et al., Science 2018): [paper](https://www.science.org/doi/10.1126/science.aar6404)
- KataGo paper (Wu 2019): [arXiv:1902.10565](https://arxiv.org/abs/1902.10565)
- KataGo follow-up methods: [`KataGoMethods.md`](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
- KataGo selfplay ops: [`SelfplayTraining.md`](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)
- Lc0 blog and wiki: [`lczero.org/blog`](https://lczero.org/blog/), [`lczero.org/dev/wiki`](https://lczero.org/dev/wiki/)
- Oracle "Lessons from AlphaZero" series by Aditya Prasad / Anthony Young (Oracle Devs Medium)
