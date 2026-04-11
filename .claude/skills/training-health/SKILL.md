---
name: training-health
description: Analyze training health — Elo progression, data quality, policy entropy, outcome distribution. Pulls live data from S3.
user_invocable: true
---

# Training Health Analysis

Pull live training data from S3 and produce a structured health report.

## How to collect data

The .env file with S3 credentials lives in the **main repo root** (not the worktree). Find it via:
```
git rev-parse --git-common-dir  # → /path/to/hexchess/.git → parent has .env
```

Use `uv run python -c '...'` snippets with `from dotenv import load_dotenv` pointed at that .env, then `from training import storage` to pull data. Collect:

### 1. Elo state
```python
state = storage.get_json(storage.ELO_STATE)
# state["ratings"] — dict of player→{mu, sigma} (OpenSkill / Plackett-Luce)
# state["pair_results"], state["player_stats"], state["total_games"], state["active_players"], state["retired_players"]
```

**IMPORTANT — do not sort by raw `mu`.** The system ranks players by a *conservative*
rating that subtracts an uncertainty penalty. Ratings are on the raw OpenSkill
scale (default μ=25, σ≈8.33) — there is no Elo rescaling.

```python
from training.elo import format_elo_table, conservative_rating, is_evaluated
# conservative_rating(mu, sigma) -> float = mu - 2 * sigma
# is_evaluated(sigma) -> bool; True iff sigma <= 2.5 (posterior tight)
print(format_elo_table(state["ratings"]))  # already sorted, marks eval/prov
```

Use `format_elo_table` (or `conservative_rating` directly) for any ranking. Sorting by
raw `mu` will rank high-uncertainty players (e.g. baselines that haven't played much)
above well-calibrated stronger models. When comparing strengths, prefer players
marked `[eval]` — a `[prov]` player's μ is not yet trustworthy.

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

### 5. Data quality (sample .npz files)
Sample ~5 .npz files evenly across selfplay data (and separately for imitation). For each:
```python
data = np.load(local_path)
outcomes = data["outcomes"]   # (N, 3) WDL from side-to-move perspective
policies = data["policies"]   # (N, ~4206) MCTS visit distribution

# Policy entropy: -sum(p * log(p)) per position
# Max probability: np.max(policies, axis=1) per position
# Non-zero actions: np.count_nonzero(policies, axis=1)
```

## How to interpret

### Elo & Strength
- Is each new model version stronger than the last? (Elo should trend upward)
- How do model versions compare to baselines (Heuristic, Minimax-2/3/4)?
- What's the draw rate trend? (Rising draws among strong players = better defensive play)

### Data Quality
- **Policy entropy**: Median near 0 = policy collapse (bad). Healthy selfplay range is 1.5-3.5.
- **Policy max probability**: If >60% of positions have max_prob > 0.9, the model is overconfident and MCTS isn't exploring.
- **Outcome distribution**: Should be roughly balanced W/D/L. Strong asymmetry = color bias.
- Compare selfplay vs imitation data quality — imitation should have higher entropy (softer targets from minimax).

### Data Volume
- Is there enough data per version? (~100K+ positions per version for meaningful training)
- Are workers active? (heartbeats within last ~15 min)

### Red Flags
- Elo regression between consecutive versions
- Policy entropy collapse (median entropy < 0.5 in selfplay)
- Models unable to beat Minimax-2 (Elo ~1500)
- Data production stalled (stale heartbeats)
- Extreme outcome imbalance (>60% one outcome)

## Output Format

Lead with a 1-line verdict (**healthy** / **concerning** / **unhealthy**), then the detailed breakdown by section.
