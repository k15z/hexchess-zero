# 03 — Leela Chess Zero: years of practical lessons

Lc0 is the longest-running open AlphaZero-style training project (started Jan 2018, still going).
Their [project history](https://lczero.org/dev/wiki/project-history/) is a goldmine — the
following is the distilled "lessons learned."

## The training run lineage

- **T1–T19** (early 2018): bootstrap, lots of bugs, learned how to do this at all.
- **T20–T30**: first credible chess play. Tuned PUCT away from early defaults (Lc0's UCI
  `CPuct` default eventually settled at 3.0 — see [lc0-options](https://lczero.org/dev/wiki/lc0-options/)).
- **T40** (late 2018): first network strong enough to beat top engines on weaker hardware.
  Experimented with dropout, learning rate decay, various policy temperatures.
- **T50** (mid 2019): introduced **WDL value head**
  ([blog](https://lczero.org/blog/2020/04/wdl-head/); all networks from July 2019 onward).
  Major quality improvement, especially for converting won endgames.
- **T60** (2020): added **moves-left head (MLH)** (project history ID 63486). Encoded
  "estimated game length" so the engine knows whether it's making progress in a winning
  position.
- **T70/T80/T82** (2021–2023): scaled to 40-block / 512-filter "BT2/BT3/BT4" networks, attention
  variants, transformer experiments.
- **v0.30 (2023)**: WDL-rescaling + contempt
  ([blog](https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/)).

## Key lessons

### 1. WDL > scalar value

Switching the value head from a single `tanh` scalar to a 3-class softmax (Win/Draw/Loss) was
worth substantial Elo, especially in endgames and drawish positions. From the
[Lc0 WDL blog](https://lczero.org/blog/2020/04/wdl-head/):

- The "+1 pawn" → "+2 pawns" jump is asymmetric: +1 is usually holdable, +2 is usually lost.
  A scalar tanh head can't represent this calibration well.
- WDL lets the engine **see drawishness** explicitly. Stockfish-style `0.00` evaluations actually
  correspond to many different W/D/L distributions — sometimes a real dead draw, sometimes
  W=0.4, D=0.2, L=0.4 (sharp, both sides have chances).
- WDL allows **contempt** to be implemented sensibly: bias choices toward higher-W/lower-D (or
  vice versa) without distorting the underlying calibration.

Use WDL. If your game has draws at all, this is non-negotiable.

### 2. Moves-left head

Without an MLH, a network in a winning position has no incentive to actually mate. It picks any
move with `W ≈ 1`, often shuffling forever. In chess this caused real problems with the 50-move
rule and threefold repetition.

The MLH outputs an estimated number of plies until game end. At search time, when two moves are
both winning, prefer the one with **fewer expected moves left** (and conversely in losing
positions, prefer more — drag it out, hope for opponent error).

This is the chess analog of KataGo's "score" target, and the principle generalizes: **give your
network a way to express urgency / progress.**

### 3. cpuct tuning is game-specific

Lc0's cpuct journey (approximate, from the project history wiki):

- The AlphaZero paper uses the dynamic form with `c1=1.25, c2=19652` (via AGZ methods), not a
  single scalar. Lc0 reuses the same `CPuctBase = 19652` constant but exposes a scalar
  `CPuct` multiplier.
- Lc0's published UCI default for `CPuct` is **3.0**, with `CPuctBase = 19652`, `CPuctFactor = 2.0`
  ([lc0-options](https://lczero.org/dev/wiki/lc0-options/)).
- Internally, training runs have used considerably lower tree `CPuct` values — e.g. T60 cycled
  through 1.3–1.9 (with `CPuctAtRoot` around 2.5), and T70+ standardised on roughly
  `cpuct 1.32, cpuctatroot 1.9`. The UCI default and the training default are not the same
  knob.
- The Oracle "Lessons from AlphaZero" series found higher `c_puct` (~4) worked well for
  Connect Four with its small branching factor
  ([Prasad blog](https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5)).

**Lesson:** never trust a `c_puct` from another paper. Tune it for your game and your visit count.
Higher `c_puct` is needed for higher visit counts; lower for lower.

### 4. FPU reduction matters more than people think

FPU = First-Play Urgency = the value assigned to an unvisited child during PUCT selection.

- AlphaZero used "loss" (i.e. `-1`), which strongly discourages exploring children with low prior.
- Lc0 found that **FPU reduction** — assigning `Q_parent − r` for a small `r` like `0.2–0.5` —
  dramatically improves search efficiency.
- For training games: `--fpu_reduction=0` (so all the exploration comes from Dirichlet noise +
  visit-prop terms).
- For match play: `0.2–0.5`.

### 5. Things that *didn't* work in Lc0

From the [project history](https://lczero.org/dev/wiki/project-history/):

- **Multinet (T58/T59)**: Training multiple nets in parallel and averaging. Slowed game
  generation badly enough to be worse on net.
- **Aggressive rule50 input plane normalization**: caused slow recovery, required reverting.
- **DTZ policy boost (T52)**: tried to use tablebase distance-to-zero info to bias policy.
  Interacted badly with KLD-based search throttling; removed.
- **Pure LR-search experiments (T55)**: declared a failure on completion.
- **Aggressive value head changes mid-run**: usually destabilized training; better to start a
  new run.

**Meta-lesson:** changing hyperparameters mid-run often costs more than the change is worth. If
you want to test a new idea seriously, start a fresh run from scratch.

### 6. Temperature decay schedule

Lc0 self-play temperature schedules have been tweaked many times. Representative values from
the T60 project-history entries include initial temperatures in the 0.9–1.3 range with a decay
over the first ~20–60 moves (e.g. "0.9 decaying to 0.45, delay-moves=20, moves=80,
cutoff-move=60"). Different from the AlphaZero "τ=1 for first 30 plies then 0" — Lc0 found
smoother decays more stable. Treat the exact numbers as run-specific; the principle is the
only durable lesson.

### 7. Contempt and WDL rescaling (v0.30)

The newest big feature: rescale the predicted WDL distribution at inference based on the player's
estimated accuracy. The point isn't strength against perfect play — it's that **a sharp
position with W=0.5/D=0.0/L=0.5 is much more dangerous against a weak opponent than a flat
W=0.0/D=1.0/L=0.0 position**, even though both are "0.50 expected." Contempt biases the engine
toward positions where the opponent is more likely to err.

For self-play training this isn't directly relevant, but it confirms that **WDL is the right
representation** — you can't do contempt cleanly with a scalar head.

## TL;DR for a new project

1. Start with WDL value head from day one. Don't migrate later.
2. Add a moves-left / progress head if your game has any drag-out problem.
3. Tune `c_puct`, `FPU reduction`, and `Dirichlet α` for *your* game empirically.
4. Don't change hyperparameters mid-run unless you're prepared to invalidate Elo comparisons.
5. Read the Lc0 history end-to-end before designing your training loop —
   [`lczero.org/dev/wiki/project-history/`](https://lczero.org/dev/wiki/project-history/).
