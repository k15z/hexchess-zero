"""Round 2: Focused experiments around hex-masked CNN.

Explores: masking, SE blocks, head sizing, pooling strategies, depth/width.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path("/Users/kevz/Desktop/hexchess/.data/benchmark_data")
EPOCHS = 5
BATCH_SIZE = 256
LR = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NUM_MOVE_INDICES = 4206
NUM_HEX_CELLS = 91
BOARD_H, BOARD_W = 11, 11
IN_CHANNELS = 16


# ---------------------------------------------------------------------------
# Hex geometry
# ---------------------------------------------------------------------------

def _hex_cells():
    cells, grid_idx = [], []
    for q in range(-5, 6):
        for r in range(-5, 6):
            if max(abs(q), abs(r), abs(q + r)) <= 5:
                cells.append((q, r))
                grid_idx.append((r + 5, q + 5))
    return cells, grid_idx

HEX_CELLS, HEX_GRID_INDICES = _hex_cells()
_HEX_ROWS = [r for r, c in HEX_GRID_INDICES]
_HEX_COLS = [c for r, c in HEX_GRID_INDICES]

def _hex_mask_2d():
    mask = torch.zeros(1, 1, 11, 11)
    for r, c in HEX_GRID_INDICES:
        mask[0, 0, r, c] = 1.0
    return mask

HEX_MASK = _hex_mask_2d()


def extract_hex(x):
    """(B, C, 11, 11) -> (B, 91, C): extract valid hex cells."""
    return x[:, :, _HEX_ROWS, _HEX_COLS].permute(0, 2, 1)


def hex_global_avg_pool(x):
    """(B, C, 11, 11) -> (B, C): average over 91 valid cells."""
    return x[:, :, _HEX_ROWS, _HEX_COLS].mean(dim=2)


def hex_global_max_pool(x):
    """(B, C, 11, 11) -> (B, C): max over 91 valid cells."""
    return x[:, :, _HEX_ROWS, _HEX_COLS].max(dim=2).values


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.conv1 = nn.Conv2d(f, f, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(f)
        self.conv2 = nn.Conv2d(f, f, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(f)

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)


class SEBlock(nn.Module):
    def __init__(self, f, ratio=4):
        super().__init__()
        mid = f // ratio
        self.fc1 = nn.Linear(f, mid)
        self.fc2 = nn.Linear(mid, 2 * f)
        self.f = f

    def forward(self, x):
        B, C, H, W = x.shape
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s))
        s = self.fc2(s)
        w, b = s[:, :C], s[:, C:]
        return torch.sigmoid(w.view(B, C, 1, 1)) * x + b.view(B, C, 1, 1)


class SEResBlock(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.conv1 = nn.Conv2d(f, f, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(f)
        self.conv2 = nn.Conv2d(f, f, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(f)
        self.se = SEBlock(f)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + x)


# ---------------------------------------------------------------------------
# Configurable model
# ---------------------------------------------------------------------------

class HexCNN(nn.Module):
    """Highly configurable hex chess CNN for benchmarking.

    Args:
        filters: conv channel width
        blocks: number of residual blocks
        use_se: use squeeze-and-excitation blocks
        hex_mask: zero out invalid cells after each block
        policy_ch: channels in policy conv head
        value_hidden: hidden dim in value head FC
        pool_mode: how to reduce spatial dims for heads
            "flatten_grid" - flatten full 11x11 (current default)
            "flatten_hex"  - flatten only 91 valid cells
            "avg_pool"     - global avg pool over valid hex cells
            "max_pool"     - global max pool over valid hex cells
            "avg_max_pool" - concat avg + max pool
    """
    def __init__(self, filters=128, blocks=6, use_se=False, hex_mask=False,
                 policy_ch=2, value_hidden=256, pool_mode="flatten_grid"):
        super().__init__()
        self.hex_mask = hex_mask
        self.pool_mode = pool_mode
        self.filters = filters

        self.input_conv = nn.Conv2d(IN_CHANNELS, filters, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(filters)

        block_cls = SEResBlock if use_se else ResBlock
        self.blocks = nn.ModuleList([block_cls(filters) for _ in range(blocks)])

        if hex_mask:
            self.register_buffer("mask", HEX_MASK)

        # Policy head
        self.policy_conv = nn.Conv2d(filters, policy_ch, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_ch)

        # Value head
        self.value_conv = nn.Conv2d(filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)

        # Compute FC input sizes based on pool mode
        if pool_mode == "flatten_grid":
            policy_in = policy_ch * BOARD_H * BOARD_W
            value_in = BOARD_H * BOARD_W
        elif pool_mode == "flatten_hex":
            policy_in = policy_ch * NUM_HEX_CELLS
            value_in = NUM_HEX_CELLS
        elif pool_mode in ("avg_pool", "max_pool"):
            policy_in = policy_ch
            value_in = 1
        elif pool_mode == "avg_max_pool":
            policy_in = policy_ch * 2
            value_in = 2
        else:
            raise ValueError(f"Unknown pool_mode: {pool_mode}")

        self.policy_fc = nn.Linear(policy_in, NUM_MOVE_INDICES)
        self.value_fc1 = nn.Linear(value_in, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def _pool(self, x):
        """Reduce (B, C, 11, 11) -> (B, C * factor)."""
        if self.pool_mode == "flatten_grid":
            return x.view(x.size(0), -1)
        elif self.pool_mode == "flatten_hex":
            return x[:, :, _HEX_ROWS, _HEX_COLS].reshape(x.size(0), -1)
        elif self.pool_mode == "avg_pool":
            return hex_global_avg_pool(x)
        elif self.pool_mode == "max_pool":
            return hex_global_max_pool(x)
        elif self.pool_mode == "avg_max_pool":
            return torch.cat([hex_global_avg_pool(x), hex_global_max_pool(x)], dim=1)

    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)))
        if self.hex_mask:
            x = x * self.mask
        for block in self.blocks:
            x = block(x)
            if self.hex_mask:
                x = x * self.mask

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_fc(self._pool(p))

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = self._pool(v)
        v = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v))))
        return p, v


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------

def load_data(max_positions=None):
    boards, policies, outcomes = [], [], []
    for f in sorted(DATA_DIR.glob("*.npz")):
        d = np.load(f)
        boards.append(d["boards"])
        policies.append(d["policies"])
        outcomes.append(d["outcomes"])
    boards = np.concatenate(boards)
    policies = np.concatenate(policies)
    outcomes = np.concatenate(outcomes)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(boards))
    boards, policies, outcomes = boards[perm], policies[perm], outcomes[perm]
    if max_positions and max_positions < len(boards):
        boards, policies, outcomes = boards[:max_positions], policies[:max_positions], outcomes[:max_positions]
    split = int(len(boards) * 0.9)
    train = TensorDataset(
        torch.from_numpy(boards[:split]),
        torch.from_numpy(policies[:split]),
        torch.from_numpy(outcomes[:split]).float(),
    )
    val = TensorDataset(
        torch.from_numpy(boards[split:]),
        torch.from_numpy(policies[split:]),
        torch.from_numpy(outcomes[split:]).float(),
    )
    print(f"Data: {split} train, {len(boards) - split} val (of {len(perm)} total)")
    return train, val


def evaluate(model, loader):
    model.eval()
    total_p, total_v, n = 0.0, 0.0, 0
    with torch.no_grad():
        for b, p, o in loader:
            b, p, o = b.to(DEVICE), p.to(DEVICE), o.to(DEVICE)
            pp, pv = model(b)
            total_p += -torch.sum(p * torch.log_softmax(pp, dim=1), dim=1).mean().item()
            total_v += F.mse_loss(pv.squeeze(-1), o).item()
            n += 1
    return total_p / n, total_v / n


def measure_inference_time(model, device, num_warmup=10, num_iters=100):
    model.eval()
    dummy = torch.randn(1, IN_CHANNELS, BOARD_H, BOARD_W, device=device)
    with torch.no_grad():
        for _ in range(num_warmup):
            model(dummy)
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        for _ in range(num_iters):
            model(dummy)
        if device.type == "mps":
            torch.mps.synchronize()
    return (time.time() - t0) / num_iters * 1000


def run_experiment(name, model, train_ds, val_ds, exp_idx, total_exps):
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"[{exp_idx}/{total_exps}] {name} ({params:,} params)")
    print(f"{'='*60}")
    model = model.to(DEVICE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    num_batches = len(train_loader)
    history = []

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for batch_idx, (b, p, o) in enumerate(train_loader):
            b, p, o = b.to(DEVICE), p.to(DEVICE), o.to(DEVICE)
            pp, pv = model(b)
            loss = -torch.sum(p * torch.log_softmax(pp, dim=1), dim=1).mean() + F.mse_loss(pv.squeeze(-1), o)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx + 1) % 200 == 0:
                print(f"    batch {batch_idx+1}/{num_batches} train_loss={running_loss/(batch_idx+1):.4f}", flush=True)
        vp, vv = evaluate(model, val_loader)
        elapsed = time.time() - t0
        history.append((vp, vv))
        print(f"  epoch {epoch+1}/{EPOCHS}: val_policy={vp:.4f} val_value={vv:.4f} total={vp+vv:.4f} [{elapsed:.1f}s]", flush=True)

    infer_ms = measure_inference_time(model, DEVICE)
    print(f"  inference: {infer_ms:.2f} ms/position")

    return {"name": name, "params": params, "history": history,
            "val_policy": history[-1][0], "val_value": history[-1][1],
            "val_total": history[-1][0] + history[-1][1], "infer_ms": infer_ms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Quick run: 10K positions, 2 epochs")
    parser.add_argument("--positions", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    if args.fast:
        max_pos = args.positions or 10_000
        epochs = args.epochs if args.epochs != 5 else 2
    else:
        max_pos = args.positions
        epochs = args.epochs

    global EPOCHS
    EPOCHS = epochs

    print(f"Device: {DEVICE}")
    print(f"Mode: {'FAST' if args.fast else 'FULL'} | {epochs} epochs | {max_pos or 'all'} positions")
    train_ds, val_ds = load_data(max_positions=max_pos)

    experiments = [
        # --- Baselines ---
        ("baseline (no mask)", HexCNN(128, 6)),
        ("hex-masked", HexCNN(128, 6, hex_mask=True)),

        # --- Hex-masked + SE ---
        ("hex-masked+SE", HexCNN(128, 6, use_se=True, hex_mask=True)),

        # --- Depth/width around hex-masked ---
        ("hex-masked 8x128", HexCNN(128, 8, hex_mask=True)),
        ("hex-masked 6x160", HexCNN(160, 6, hex_mask=True)),

        # --- Policy head sizing ---
        ("hex-masked pol-8ch", HexCNN(128, 6, hex_mask=True, policy_ch=8)),
        ("hex-masked pol-16ch", HexCNN(128, 6, hex_mask=True, policy_ch=16)),

        # --- Value head sizing ---
        ("hex-masked val-512", HexCNN(128, 6, hex_mask=True, value_hidden=512)),

        # --- Pooling strategies (all hex-masked) ---
        ("hex-masked flat-hex", HexCNN(128, 6, hex_mask=True, pool_mode="flatten_hex")),
        ("hex-masked avg-pool", HexCNN(128, 6, hex_mask=True, pool_mode="avg_pool")),
        ("hex-masked max-pool", HexCNN(128, 6, hex_mask=True, pool_mode="max_pool")),
        ("hex-masked avg+max", HexCNN(128, 6, hex_mask=True, pool_mode="avg_max_pool")),

        # --- Combined best ideas ---
        ("hex+SE+pol8+flat-hex", HexCNN(128, 6, use_se=True, hex_mask=True, policy_ch=8, pool_mode="flatten_hex")),
        ("hex+SE+pol8+avg+max", HexCNN(128, 6, use_se=True, hex_mask=True, policy_ch=8, pool_mode="avg_max_pool")),
    ]

    results = []
    total = len(experiments)
    t_start = time.time()
    for i, (name, model) in enumerate(experiments):
        r = run_experiment(name, model, train_ds, val_ds, i + 1, total)
        results.append(r)
        elapsed_total = time.time() - t_start
        remaining = (elapsed_total / (i + 1)) * (total - i - 1)
        print(f"  [{i+1}/{total} done, ~{remaining/60:.0f}m remaining]", flush=True)

    # Summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY (sorted by total val loss, lower is better)")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Params':>10} {'Policy':>8} {'Value':>8} {'Total':>8} {'ms/pos':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["val_total"]):
        print(f"{r['name']:<30} {r['params']:>10,} {r['val_policy']:>8.4f} {r['val_value']:>8.4f} {r['val_total']:>8.4f} {r['infer_ms']:>7.2f}")

    print(f"\n{'='*80}")
    print("CONVERGENCE (val_total by epoch)")
    print(f"{'='*80}")
    header = f"{'Model':<30}" + "".join(f"{'E'+str(i+1):>9}" for i in range(EPOCHS))
    print(header)
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["val_total"]):
        row = f"{r['name']:<30}"
        for vp, vv in r["history"]:
            row += f"{vp+vv:>9.4f}"
        print(row)


if __name__ == "__main__":
    main()
