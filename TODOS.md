# TODOs

## Search and model

- **Revisit the attention policy head with the real move table** — the earlier experiment used a placeholder move-index mapping and underperformed. If this comes back, it should be wired to the engine's actual `move_to_index()` table rather than sequential cell pairs.
- **Smolgen-style dynamic position encoding** — Lc0's chess-specific idea: compress the board state into a vector and generate supplemental attention logits per head. Potentially useful if we revisit more attention-heavy policy heads.
- **Tree reuse across moves** — keep the subtree rooted at the chosen move instead of rebuilding from scratch each turn.

## Training pipeline

- **Surprise weighting for replay sampling** — upweight positions where the raw network disagrees with the final MCTS target (policy KL / value error) and carry those weights through replay.

## Testing and API ergonomics

- **Property-based tests** — fuzz move generation, notation round-trips, and apply/undo invariants across randomized legal positions.
- **Evaluator hot-path cleanup** — profile whether the current `(Vec<f32>, f32)` evaluator API is meaningfully allocating in search, and switch to a caller-owned policy buffer only if the profiler says it matters.
