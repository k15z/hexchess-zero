# Training Health Analysis

Use this runbook when asked to analyze training health, especially Elo progression, data quality, policy entropy, and outcome distribution from live S3 data.

## How to collect data

The `.env` file with S3 credentials lives in the **main repo root** (not the worktree). Find it via:

```bash
git rev-parse --git-common-dir
```

Use `uv run python -c '...'` snippets with `from dotenv import load_dotenv` pointed at that `.env`, then `from training import storage` to pull data.

### 1. Elo state

```python
state = storage.get_json(storage.ELO_STATE)
# state["ratings"] - dict of player->{mu, sigma} (OpenSkill / Plackett-Luce)
# state["pair_results"], state["player_stats"], state["total_games"], state["active_players"], state["retired_players"]
```

Important: do not sort by raw `mu`. The system ranks players by a conservative rating that subtracts an uncertainty penalty. Ratings are on the raw OpenSkill scale (default `mu=25`, `sigma~=8.33`) - there is no Elo rescaling.

```python
from training.elo import format_elo_table, conservative_rating, is_evaluated

# conservative_rating(mu, sigma) -> float = mu - 2 * sigma
# is_evaluated(sigma) -> bool; True iff sigma <= 2.5 (posterior tight)
print(format_elo_table(state["ratings"]))  # already sorted, marks eval/prov
```

Use `format_elo_table` or `conservative_rating` directly for ranking. Sorting by raw `mu` can place high-uncertainty baselines above calibrated stronger models. When comparing strengths, prefer players marked `[eval]`.

Sigma regimes at a glance:

- `sigma > 4`: matchmaker treats as high uncertainty and prioritizes placement pairings (`_select_pair` in `elo_service.py`). Displayed as `[prov]`.
- `2.5 < sigma <= 4`: posterior is narrowing but `mu` still has meaningful error bars. Still `[prov]`.
- `sigma <= 2.5`: `[eval]`. Posterior is tight; `mu - 2*sigma` is a defensible lower bound.

### 2. Model versions

```python
meta = storage.get_json(storage.LATEST_META)  # {"version": N, "timestamp": "..."}
version_keys = storage.ls(storage.VERSIONS_PREFIX)  # models/versions/*.onnx
```

### 3. Data volume

```python
selfplay_files = storage.list_data_files(storage.SELFPLAY_PREFIX)
imitation_files = storage.list_data_files(storage.IMITATION_PREFIX)
# Each entry: {"key": str, "positions": int, "timestamp": str}
# Group selfplay by version: key.split("/")[2] gives "v1", "v2", etc.
```

### 4. Worker heartbeats

```python
for key in storage.ls(storage.HEARTBEATS_PREFIX):
    hb = storage.get_json(key)  # {timestamp, version, total_games, total_positions}
```

### 5. Data quality

Sample roughly 5 `.npz` files evenly across self-play data, and separately across imitation data. For each:

```python
data = np.load(local_path)
outcomes = data["outcomes"]   # (N, 3) WDL from side-to-move perspective
policies = data["policies"]   # (N, ~4206) MCTS visit distribution

# Policy entropy: -sum(p * log(p)) per position
# Max probability: np.max(policies, axis=1) per position
# Non-zero actions: np.count_nonzero(policies, axis=1)
```

## How to interpret

### Elo and strength

- Check whether each new model version is stronger than the last.
- Compare model versions to baselines such as Heuristic and Minimax-2/3/4.
- Track draw rate trends. Rising draws among strong players can indicate better defensive play.

### Data quality

- Policy entropy: median near 0 indicates policy collapse. Healthy self-play is usually in the `1.5-3.5` range.
- Policy max probability: if more than 60% of positions have `max_prob > 0.9`, the model is likely overconfident and MCTS is not exploring enough.
- Outcome distribution: should be roughly balanced W/D/L. Strong asymmetry suggests color bias.
- Compare self-play to imitation. Imitation should generally have higher entropy.

### Data volume

- Check for enough data per version, roughly `100K+` positions for meaningful training signals.
- Check for active workers by ensuring heartbeats are fresh, roughly within the last 15 minutes.

### Red flags

- Elo regression between consecutive versions
- Policy entropy collapse, especially median entropy `< 0.5` in self-play
- Models unable to beat Minimax-2
- Data production stalled because heartbeats are stale
- Extreme outcome imbalance, especially `>60%` concentrated in one outcome

## Output format

Lead with a one-line verdict: `healthy`, `concerning`, or `unhealthy`, then provide the detailed breakdown by section.
