# Training Health Analysis

Use this runbook when asked to analyze live training health. The focus is the current self-play pipeline plus the evaluation path that decides whether a candidate should replace the approved model.

## How to collect data

The `.env` file with S3 credentials lives in the main repo root, not necessarily the worktree root. Find it with:

```bash
git rev-parse --git-common-dir
```

Use `uv run python -c '...'` snippets with `from dotenv import load_dotenv` pointed at that `.env`, then import `training.storage`.

### 1. Model pointers

```python
latest = storage.get_json(storage.LATEST_META)
approved = storage.get_json(storage.APPROVED_META)
```

Important:

- `latest` is the newest exported trainer candidate.
- `approved` is the model workers are actually using for self-play.
- If `latest["version"] > approved["version"]`, a candidate is pending evaluation or has been rejected.

### 2. Evaluation artifacts

List evaluated versions and inspect the newest one:

```python
versions = storage.list_eval_versions()
target = max(versions)
gate = storage.get_json(storage.eval_gate_summary_key(target))
bench = storage.get_json(storage.eval_benchmark_summary_key(target))
decision = storage.get_json(storage.eval_decision_key(target))
records = [
    storage.get_json(key)
    for key in storage.list_eval_game_record_keys(target)[-20:]
]
```

Key fields to inspect:

- `gate["status"]`, `gate["sprt"]`, `gate["pair_buckets"]`
- `bench["status"]`, `bench["per_opponent"]`
- `decision["status"]`, `decision["reasons"]`
- per-game `eval_type`, `pair_id`, `pair_game_index`, `opening_seed`, `opening_plies`

### 3. Trainer and data volume

```python
trainer = storage.get_json(storage.TRAINER_METRICS)
selfplay_files = storage.list_data_files(storage.SELFPLAY_PREFIX)
imitation_files = storage.list_data_files(storage.IMITATION_PREFIX)
```

Useful checks:

- cumulative self-play volume
- fresh self-play since `approved["positions_at_promote"]` when present
- whether trainer summaries are still advancing

### 4. Worker heartbeats

```python
heartbeats = {
    key: storage.get_json(key)
    for key in storage.ls(storage.HEARTBEATS_PREFIX)
}
```

Look for:

- stale workers
- workers on an old approved version
- flatlined `total_games` / `total_positions`

### 5. Data quality

Sample a few `.npz` files from self-play and imitation:

```python
data = np.load(local_path)
outcomes = data["outcomes"]
policies = data["policies"]
```

Check:

- policy entropy
- max policy probability
- non-zero action counts
- outcome balance

## How to interpret

### Evaluation status

- `decision["status"] == "promoted"`: healthy candidate promotion path.
- `decision["status"] == "pending"`: inspect whether gate evidence or anchor evidence is the bottleneck.
- `decision["status"] == "rejected_pending"`: the candidate is already non-promotable, but the service is still finishing the scheduled gate / benchmark evidence.
- `decision["status"] == "rejected"`: identify whether the rejection came from direct gate failure or anchor regression.
- `decision["status"] == "baseline_ready"`: the approved model benchmark baseline exists, but no newer candidate is under review yet.

### Gate quality

Prefer the paired-color view over raw win rate:

- check `pair_buckets` and color split, not just `wins / losses / draws`
- check whether both halves of each pair exist
- check that `opening_seed` varies across pairs
- check whether one color is carrying all of the score

### Benchmark regressions

For each anchor:

- compare `score` to `target_score`
- inspect `sprt.status`
- inspect `termination_mix` and `move_time_stats`

If only one anchor fails, that is usually more actionable than a global “candidate bad” label.

### Training health

Red flags:

- fresh self-play is not increasing
- trainer summaries are stale
- workers are missing or behind the approved pointer
- policy entropy collapses
- outcome distribution becomes heavily color-skewed
- evaluation games show abnormal termination mix or timing spikes

## Output format

Lead with one verdict:

- `healthy`
- `concerning`
- `unhealthy`

Then summarize:

1. model pointers and fresh-position cadence
2. current evaluation state
3. worker / trainer health
4. data-quality findings
