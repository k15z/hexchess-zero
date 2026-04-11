# Storage, Dashboard & Deployment Review (2026-04-11)

## Findings

### 1) Raw S3 key string outside `training/storage.py` (policy violation)
- `training/gating.py` defines `GATE_STATE_KEY = "state/gate.json"` directly instead of importing a constant from `training/storage.py`.
- This violates the repository convention that S3 keys should be centralized in `training/storage.py`.
- **Risk:** key drift and subtle breakage if prefixes are refactored.
- **Recommendation:** add `GATE_STATE = "state/gate.json"` in `training/storage.py` and consume it from `training/gating.py`.

### 2) Startup resilience gap in dashboard sync loop
- `DashboardStore.start()` calls `refresh_once()` synchronously before spawning the background thread.
- If S3 is briefly unavailable at startup, this exception bubbles up and can prevent dashboard boot.
- The background loop itself is resilient (`_loop` catches exceptions and keeps retrying), but that protection does not apply to the initial prime.
- **Risk:** transient outage causes crash-loop rather than degraded startup.
- **Recommendation:** wrap startup `refresh_once()` with retry/backoff or allow start with `initialised=False` + `sync_error`.

### 3) Sim-count drift vs current operational guidance
- `training/config.py` currently defaults worker simulations to 200.
- `k8s/worker.yaml` comments still describe 500 sims and CPU tuning around that value.
- Repository guidance in `CLAUDE.md` requires MCTS/eval workflows to use >=800 simulations.
- **Risk:** inconsistent quality/performance expectations and misleading capacity planning docs.
- **Recommendation:** standardize worker sim target (code defaults, compose/k8s args, and comments) and retune resource requests with measured throughput at the chosen sim count.

### 4) Deployment drift: local compose vs k8s image sourcing
- `docker-compose.yml` uses `build: .` for all services (local image), while k8s manifests pin `ghcr.io/...:latest`.
- **Risk:** local verification may run different bits than cluster deployments.
- **Recommendation:** consider explicit image tags in both paths or document expected divergence clearly.

## Checks that look good

- **S3 pagination:** `training/storage.py` uses `list_objects_v2` with continuation tokens in both `ls` and `list_with_meta`.
- **Dashboard elo_games incremental sync:** tail-window diff logic handles new/deleted tail keys by recomputing cached tail set and fetching only unseen objects.
- **Transient sync errors after startup:** background loop catches exceptions and records `sync_error`, then continues retrying next interval.
- **`.env` handling:** `.env` is gitignored; compose references `.env` via `env_file` without code paths that directly print S3 credential values.
- **Docker build layering:** Rust/PyO3 wheel is built in a dedicated builder stage over Rust sources; Python-only edits under `training/` should avoid rebuilding Rust wheel layers when cache is warm.
- **Health checks wiring:** `run_all_invariants` is actively invoked during trainer startup, and runtime checks are executed periodically.

## Notes on ETag semantics (S3/R2/Spaces)

- Using ETag for change detection is generally safe as a **cache invalidation hint**, but ETag is not a universal content hash across providers and multipart upload patterns.
- In this code path, an ETag change causes an extra GET (safe), while unchanged ETag skips GET.
- Practical risk here is mostly efficiency false-positives (extra fetches), not stale reads, assuming normal object-store behavior for overwrite/list consistency in the target backend.
