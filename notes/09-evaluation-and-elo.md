# 09 — Evaluation and Elo

## Why this is hard

Self-play Elo is a **rubber ruler**. If your only opponent is your own previous network, you can
only measure relative progress within the closed pool. Two well-known failure modes:

1. **Elo inflation**: closed pools tend to inflate ratings because each new model exploits its
   predecessors' weaknesses (relative gain) without necessarily playing globally better chess
   (absolute gain). Lc0 has documented this in many training runs.
2. **Cycles**: A beats B, B beats C, C beats A. Common when training pushes the population around
   a local optimum.

The fix: **always have at least one external anchor that doesn't change.**

## External anchors

Pick one or more fixed opponents that span a wide skill range:

- **Random player** — sanity check the bottom of the scale.
- **Material-only minimax depth-1** — checks basic tactical awareness.
- **Material + mobility minimax depth-N** for several `N` — gives a Elo curve.
- **A frozen NN snapshot** from earlier in your run — measures absolute progress.

For chess: external engines (Stockfish at fixed nodes, Komodo, etc.) at multiple skill levels.
For Go: GnuGo, Pachi, frozen Leela versions.

## Rating systems

- **Bayeselo** ([remi-coulom.fr/Bayesian-Elo](https://www.remi-coulom.fr/Bayesian-Elo/)) — the
  classic. Maximum a posteriori Elo ratings from a list of game results.
- **Ordo** ([github.com/michiguel/Ordo](https://github.com/michiguel/Ordo)) — improved Bayeselo,
  better numerical stability.
- **OpenSkill / Plackett-Luce / Weng-Lin** — Bayesian rating with explicit uncertainty.
  Better than Elo for sparse round-robins; rescale to look Elo-like by `μ = 1500 + 173.7178 · θ`.
  This is what the hex-chess project uses.
- **TrueSkill** — Microsoft's rating system, similar uncertainty handling.

For training pipelines where new models constantly enter and old ones don't play many games,
**use a Bayesian rating system with uncertainty intervals** rather than naive Elo. Plot the
rating ± 1.96σ over time.

## Match formats

### Round-robin

Every model plays every other model `k` games. Quadratic in number of models. Use for small
populations (e.g. last 10 versions).

### Gauntlet

A new model plays a fixed set of "gauntlet" opponents. Linear in number of models. Use for
continuous rating updates.

### SPRT (Sequential Probability Ratio Test)

When you want to test "is model A meaningfully stronger than model B?":

- Hypothesis: A's true Elo gain over B is at least `Elo0` (e.g. +5).
- Null: gain is at most `Elo1` (e.g. 0).
- Play games until the likelihood ratio crosses one of two thresholds.
- Average ~1000–4000 games for typical ±5 Elo decisions.

The Stockfish fishtest framework popularized SPRT for chess engine development. SPRT is the
right call for **gating decisions** ("is this new feature actually an improvement?") but
unnecessary for continuous rating tracking.

### Predict-draw matchmaking

For an Elo service that has limited compute per cycle: pick pairings that maximize *expected
information* — i.e. games most likely to be informative because the rating difference is small
or uncertain. KataGo's distributed training and the hex-chess project both use this.

## Game count and noise

Rule of thumb (from chess engine testing):

| Games | Approximate ±1σ noise on Elo difference |
|---|---|
| 100 | ±35 Elo |
| 400 | ±18 Elo |
| 1000 | ±11 Elo |
| 4000 | ±5.5 Elo |
| 16000 | ±2.8 Elo |

To distinguish a 10-Elo improvement at 95% confidence, you need ~1000+ games. To distinguish a
3-Elo improvement, ~16k+. This is why fishtest is a giant distributed cluster.

For self-play training, you don't usually care about <10 Elo decisions, so 200–500 games per
matchup is enough.

## Avoiding gating disasters

If you decide to gate (require a new candidate to beat the current model by N% in M games before
promotion):

- **Use a low threshold (52–54%, not 55%)** with enough games (~400+) to make it statistically
  meaningful.
- **Beware deadlock**: if your nets are very close in strength, you can get stuck for a long
  time waiting for a candidate to clear the gate. KataGo and recent Lc0 just don't gate.
- **Always have an escape hatch**: if N candidates in a row fail to gate, promote anyway.

## Production checklist

A reasonable Elo service for an AlphaZero-style training run:

1. **Anchor**: a frozen `model_v0`, plus a fixed minimax baseline at depth 3 and depth 6.
2. **Pool**: last ~20 model versions.
3. **Matchmaker**: pick pairings by max-uncertainty / predict-draw.
4. **Engine**: ≥800 MCTS sims per move (lower numbers give misleading results — see CLAUDE.md).
5. **Time**: ~30–60 seconds per game on whatever hardware you have.
6. **Storage**: persist each game outcome immutably (timestamp, both models, result, opening).
   Recompute the rating projection from the log on demand.
7. **Dashboard**: plot rating ± 1σ over time, draw rate, average game length, anchor anchor
   anchor.

The "store immutable game records, recompute projection" pattern (vs. updating a shared rating
state) is much more robust to bugs and supports horizontal scaling.

## Sanity checks

- **Anchor regression**: if your latest model loses ground against the random / minimax-d3
  anchor, your training has gone wrong even if internal Elo says it's improving.
- **Draw rate explosion**: if both sides start drawing >70% of games, your training is
  collapsing toward "play-it-safe" (see `10-common-failure-modes.md`).
- **Anchor with large `c_puct` swing**: rerun the eval with a different `c_puct` setting; if the
  rating moves dramatically, your engine is over-fit to one search configuration.
