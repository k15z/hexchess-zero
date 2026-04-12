# 10 — Common failure modes (the bug runbook)

A catalogue of bugs and pathologies people have hit in AlphaZero-style training, with symptoms
and fixes. Sourced from: KataGo issues, Lc0 wiki / Discord summaries, the Oracle "Lessons from
AlphaZero" series, and OpenSpiel/Minigo discussions.

## 1. Policy collapse

**Symptom:** Policy entropy crashes to near-zero very early. Network plays the same move from
every position. Training loss decreases, but Elo plateaus or drops.

**Causes:**
- Dirichlet `α` too small or noise weight `ε` too small — no exploration at the root.
- `c_puct` too low — search exploits the network's overconfident prior, never visits other moves.
- Temperature decayed too fast, so all training games are deterministic.
- LR too high in early training: the policy head saturates before MCTS can correct it.

**Fixes:**
- Raise `α` toward the `10/avg_legal_moves` heuristic.
- Raise `ε` to 0.25 (the AZ default) or higher early in training.
- Raise `c_puct` and add FPU reduction.
- Keep `τ=1` for at least the first 10–30 plies of every training game.
- Lower LR or warm up.

## 2. Value head saturating to ±1

**Symptom:** All value predictions are ≈+1 or ≈−1; value loss is high; entropy of WDL is
near-zero on positions that are clearly drawn.

**Causes:**
- `tanh` scalar saturating after seeing too many won/lost terminal labels.
- Class imbalance: training data is mostly W/L with very few D positions, network learns to
  ignore the draw class.
- BatchNorm in train-mode at inference (statistics from training don't generalize).

**Fixes:**
- Switch to WDL cross-entropy (always).
- Resample to balance W/D/L if your draw rate is very low or very high.
- Verify you're calling `model.eval()` during inference.
- Lower the per-sample weight on terminal-only value loss; add a short-horizon aux value head
  (KataGo style) to give denser supervision.

## 3. Draw rate exploding

**Symptom:** After a while, >80% of self-play games are draws. Training stalls because there's
no signal in the value targets.

**Causes:**
- Network has converged to a "play it safe" strategy and there's no exploration to break out.
- No moves-left-head, so the engine has no incentive to convert won positions.
- Dirichlet noise too weak for the current branching factor.
- Self-play games have no temperature in the opening — same opening every game.

**Fixes:**
- Add MLH or equivalent progress signal (KataGo's score / Lc0's MLH).
- Increase opening temperature or use a small "opening book" of forced random first moves.
- Increase Dirichlet noise.
- Lc0 has hit this multiple times; the standard fix is more exploration plus MLH.

## 4. Repetition / shuffle loops

**Symptom:** Training games end at the move limit because both sides shuffle pieces back and
forth without ever resolving the position.

**Causes:**
- Network is in a winning position but has no incentive to make progress (no MLH).
- Repetition detection in your engine is broken — positions are being treated as new each time.
- Search treats 2-fold as not-yet-draw and wastes sims on repeating cycles.

**Fixes:**
- Add MLH.
- Treat 2-fold within the same search subtree as a draw (KataGo + Lc0 do this).
- Force draw on threefold and stop the game immediately in self-play.

## 5. Stale replay buffer / training-on-old-data

**Symptom:** Training loss decreases but live Elo against fixed baseline doesn't improve.
Sometimes Elo *regresses*.

**Causes:**
- Replay window too large — most of the buffer is from a much weaker network.
- Sample reuse rate too high — overfitting to the same data.
- Trainer running far faster than self-play — same data resampled many times.

**Fixes:**
- Use a sublinear-growing window (KataGo formula, see `06-training-loop.md`).
- Cap sample reuse at ~4–8.
- Throttle the trainer with a token bucket so it can only consume new data as fast as
  self-play produces it. KataGo and Hexchess Zero both do this.

## 6. NaN losses

**Symptom:** After some number of steps, loss becomes NaN.

**Causes:**
- LR too high.
- Unmasked illegal-move logits left as `-inf`, sometimes flowing into a softmax that produces
  NaN gradients.
- BatchNorm with batch size 1 in some pathological code path.
- log(0) somewhere — typically a `log(p)` on a `p` that's exactly zero due to FP underflow.

**Fixes:**
- Gradient clipping (norm 1.0 or 5.0).
- Lower LR / warmup.
- Use `−1e9` instead of `−inf` for masked logits.
- Add `eps` to all logs: `log(p + 1e-9)`.

## 7. BatchNorm in eval mode

**Symptom:** Network plays well in training (when BN uses running stats from train mode) but
plays terribly in inference. Or vice versa.

**Causes:** Forgetting to call `.eval()` / `.train()` at the right times. PyTorch's BN tracks
running stats only when `model.training == True`.

**Fixes:**
- Always call `model.eval()` before exporting to ONNX.
- Verify by running the same position through PyTorch and ONNX, checking outputs match within
  ~1e-5.
- KataGo eventually moved to **no batch norm** (Fixup-style init + variance preservation) to
  avoid this entire class of bug.

## 8. Move-encoding off-by-one

**Symptom:** Policy loss is much higher than `log(num_legal)`. Network sometimes outputs high
probability on illegal moves.

**Causes:** Bug in the move-to-index function. Most insidious when promotions, en passant, or
hex-coordinate edge cases are mis-encoded.

**Fixes:**
- Unit-test the move bijection: encode → decode → encode for every legal move from a sample of
  positions, verify round-trip.
- Verify the `encode_board → predict → decode → make_move` pipeline produces only legal moves.
- This bug, once shipped to production, **invalidates all training data** with the wrong
  encoding. Fix carefully.

## 9. Symmetry / orientation bugs

**Symptom:** Network plays much better as one color than the other, or much better in
top-of-board positions than bottom.

**Causes:**
- Forgetting to flip the board to "side-to-move's perspective" before encoding.
- Pawn direction encoded inconsistently between training and inference.
- Hex coordinate transform applied in one place but not another.

**Fixes:**
- **Always present positions to the network in side-to-move's frame.**
- Test by feeding mirror positions and verifying the policy is mirrored.

## 10. MCTS exploration collapse

**Symptom:** MCTS at the root spends >95% of its visits on one move from move 1, even with
Dirichlet noise.

**Causes:**
- FPU set to "loss" — every unvisited child is `Q=−1`, never gets explored.
- `c_puct` very low.
- Dirichlet noise weight too low.

**Fixes:** see "Policy collapse" above. Often the same root cause.

## 11. Overfitting to recent games

**Symptom:** Loss is good, network beats other recent versions, but loses to older versions.

**Causes:** Replay window too small or too aggressive sample reuse. Training has memorized
recent self-play patterns at the expense of general play.

**Fixes:** Larger window. Add weight decay. Lower sample reuse.

## 12. Bootstrap-from-random getting stuck

**Symptom:** Random network → random self-play → no signal → loss flat for many hours.

**Causes:** Pure cold-start can take a long time on small games where most positions are
"don't-blunder-the-king" patterns the random network never finds.

**Fixes:**
- Imitation bootstrap from minimax (recommended).
- Lower MCTS sims for the first ~hour, then ramp up — random net + 800 sims is wasted compute.
- Increase Dirichlet noise to ~0.5 for the first hour.

## 13. Underflow in policy softmax over many moves

**Symptom:** With ~4000 move indices and a confident network, most outputs underflow to 0,
training cross-entropy returns NaN.

**Causes:** FP16 / mixed-precision softmax overflow.

**Fixes:** Compute softmax in FP32. Use `log_softmax` + `nll_loss` instead of `softmax + log`.

## 14. Off-by-one in repetition counting

**Symptom:** Engine accepts draw on the third *visit* not the third *repetition*.

**Causes:** Confusing position-count with repetition-count.

**Fixes:** Test against known repetition sequences. Hexchess Zero encodes repetition
count in an input plane — make sure this matches what your training data expects.

## 15. Trainer / selfplay LR mismatch

**Symptom:** Trainer keeps progressing on loss but self-play workers have an old model that
disagrees more and more with the trainer's current policy. Eventually the model promotion
includes a discontinuous jump.

**Causes:** Promotion frequency too low.

**Fixes:** Promote every 200k–1M training samples. Workers should poll for new models every
few minutes.

---

## Meta-principle: small changes only mid-run

Almost every "I broke my training run" story in Lc0 history involves changing a hyperparameter
mid-run. **Don't.** Start a new run if you want to test a non-trivial change. Mid-run, only change
things that are clearly safe (LR drops, longer runs of the same).
