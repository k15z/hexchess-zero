# TODOs

## Can do now

- **Tree reuse across moves** — keep the subtree rooted at the chosen move instead of rebuilding from scratch each turn.
- **Enable surprise weighting for replay sampling** — the plumbing exists (sample_weight is threaded through losses, `policy_kl_target_vs_prior` is in the data schema). Just needs to be toggled on: upweight positions where the raw network disagrees with the final MCTS target.
- **Docker image tagging and config-driven experiments** — add a tagging scheme for training Docker images (build hash, experiment ID) and wire image tags into experiment configs so runs are reproducible and pinned to specific builds.
- **Integrate attention policy head into production** — bench experiments validated it with the real move table (2.4% loss improvement, same param budget). Remaining work is merging `AttentionPolicyHead` from experiments into `model.py` and retraining.
- **Property-based tests** — fuzz move generation, notation round-trips, and apply/undo invariants across randomized legal positions.

## Do when we wipe S3

- **Switch to UUIDs for IDs** — replace sequential/string IDs for `game_id`, `instance_id`, `elo_game_run_id`, etc. with UUIDs to avoid collisions and simplify distributed coordination.
