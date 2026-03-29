"""Round 3: Head-to-head — hex-masked+SE baseline vs attention policy head.

Compares the best architecture from rounds 1-2 (hex-masked SE-ResNet 6x128
with conv policy head) against a variant using an attention-based policy head
inspired by Lc0.
"""

import argparse
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

def _hex_grid_indices():
    indices = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            if max(abs(q), abs(r), abs(q + r)) <= 5:
                indices.append((r + 5, q + 5))
    return indices

HEX_GRID_INDICES = _hex_grid_indices()
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


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    def __init__(self, f, ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(f, f // ratio)
        self.fc2 = nn.Linear(f // ratio, 2 * f)

    def forward(self, x):
        B, C, H, W = x.shape
        s = F.relu(self.fc1(x.mean(dim=(2, 3))))
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


class HexMaskedSETrunk(nn.Module):
    """Shared trunk: input conv + hex-masked SE residual blocks."""
    def __init__(self, filters=128, blocks=6):
        super().__init__()
        self.input_conv = nn.Conv2d(IN_CHANNELS, filters, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(filters)
        self.blocks = nn.ModuleList([SEResBlock(filters) for _ in range(blocks)])
        self.register_buffer("mask", HEX_MASK)

    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x))) * self.mask
        for block in self.blocks:
            x = block(x) * self.mask
        return x


# ---------------------------------------------------------------------------
# Value head (shared between both models)
# ---------------------------------------------------------------------------

class ConvValueHead(nn.Module):
    def __init__(self, filters, hidden=256):
        super().__init__()
        self.conv = nn.Conv2d(filters, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(BOARD_H * BOARD_W, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        v = F.relu(self.bn(self.conv(x)))
        v = v.view(v.size(0), -1)
        return torch.tanh(self.fc2(F.relu(self.fc1(v))))


# ---------------------------------------------------------------------------
# Model A: Conv policy head (current best from round 2)
# ---------------------------------------------------------------------------

class ConvPolicyHead(nn.Module):
    def __init__(self, filters, policy_ch=2):
        super().__init__()
        self.conv = nn.Conv2d(filters, policy_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(policy_ch)
        self.fc = nn.Linear(policy_ch * BOARD_H * BOARD_W, NUM_MOVE_INDICES)

    def forward(self, x):
        p = F.relu(self.bn(self.conv(x)))
        return self.fc(p.view(p.size(0), -1))


class HexSE_ConvPolicy(nn.Module):
    """Best from round 2: hex-masked SE trunk + conv policy head."""
    def __init__(self, filters=128, blocks=6):
        super().__init__()
        self.trunk = HexMaskedSETrunk(filters, blocks)
        self.policy = ConvPolicyHead(filters)
        self.value = ConvValueHead(filters)

    def forward(self, x):
        x = self.trunk(x)
        return self.policy(x), self.value(x)


# ---------------------------------------------------------------------------
# Model B: Attention policy head
# ---------------------------------------------------------------------------

def _build_move_table():
    """Build (from_cell_idx, to_cell_idx) for each of 4206 move indices.

    We import hexchess to get the authoritative move table, then map
    each move's (from_q, from_r) and (to_q, to_r) to hex cell indices.
    """
    try:
        import hexchess
        cell_to_idx = {c: i for i, c in enumerate(
            [(q, r) for q in range(-5, 6) for r in range(-5, 6)
             if max(abs(q), abs(r), abs(q + r)) <= 5]
        )}
        n = hexchess.num_move_indices()
        from_indices = []
        to_indices = []
        for i in range(n):
            m = hexchess.index_to_move(i)
            fq, fr = m["from_q"], m["from_r"]
            tq, tr = m["to_q"], m["to_r"]
            from_indices.append(cell_to_idx[(fq, fr)])
            to_indices.append(cell_to_idx[(tq, tr)])
        return torch.tensor(from_indices, dtype=torch.long), torch.tensor(to_indices, dtype=torch.long)
    except ImportError:
        raise ImportError("hexchess bindings required for attention policy head")

_MOVE_FROM, _MOVE_TO = _build_move_table()


class AttentionPolicyHead(nn.Module):
    """Lc0-style attention policy head.

    Instead of 4206 query vectors (too much memory), we:
    1. Run a small self-attention over the 91 hex cell features
    2. For each move, gather the from-cell and to-cell features
    3. Dot product + bias to produce the move logit

    This is O(91^2) attention instead of O(4206 * 91), and the
    move logits are computed via cheap gather + dot product.
    """
    def __init__(self, filters=128, nhead=4, attn_layers=1, head_dim=64):
        super().__init__()
        self.head_dim = head_dim

        # Project trunk features for attention
        self.embed = nn.Linear(filters, head_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, NUM_HEX_CELLS, head_dim) * 0.02)

        # Self-attention over 91 cells (cheap: 91x91 attention matrix)
        layer = nn.TransformerEncoderLayer(
            d_model=head_dim, nhead=nhead, dim_feedforward=head_dim * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=attn_layers)

        # From/to projections for move logit computation
        self.from_fc = nn.Linear(head_dim, head_dim)
        self.to_fc = nn.Linear(head_dim, head_dim)

        # Per-move bias (small: just 4206 scalars)
        self.move_bias = nn.Parameter(torch.zeros(NUM_MOVE_INDICES))

        # Register move table as buffers
        self.register_buffer("move_from", _MOVE_FROM)
        self.register_buffer("move_to", _MOVE_TO)

    def forward(self, x):
        # x: (B, C, 11, 11) -> extract 91 cells -> (B, 91, C)
        feats = extract_hex(x)
        feats = self.embed(feats) + self.pos_embed  # (B, 91, head_dim)
        feats = self.attn(feats)  # (B, 91, head_dim)

        B = feats.size(0)

        # Project from/to features
        from_feats = self.from_fc(feats)  # (B, 91, head_dim)
        to_feats = self.to_fc(feats)      # (B, 91, head_dim)

        # Gather from/to features for each move
        f = from_feats[:, self.move_from]  # (B, 4206, head_dim)
        t = to_feats[:, self.move_to]      # (B, 4206, head_dim)

        # Move logit = dot(from, to) + bias
        logits = (f * t).sum(dim=-1) + self.move_bias  # (B, 4206)
        return logits


class HexSE_AttnPolicy(nn.Module):
    """Hex-masked SE trunk + attention policy head."""
    def __init__(self, filters=128, blocks=6, nhead=4, attn_layers=1, head_dim=64):
        super().__init__()
        self.trunk = HexMaskedSETrunk(filters, blocks)
        self.policy = AttentionPolicyHead(filters, nhead=nhead, attn_layers=attn_layers, head_dim=head_dim)
        self.value = ConvValueHead(filters)

    def forward(self, x):
        x = self.trunk(x)
        return self.policy(x), self.value(x)


# ---------------------------------------------------------------------------
# Training / eval (same as bench_architectures.py)
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


def evaluate(model, loader, dev):
    model.eval()
    total_p, total_v, n = 0.0, 0.0, 0
    with torch.no_grad():
        for b, p, o in loader:
            b, p, o = b.to(dev), p.to(dev), o.to(dev)
            pp, pv = model(b)
            total_p += -torch.sum(p * torch.log_softmax(pp, dim=1), dim=1).mean().item()
            total_v += F.mse_loss(pv.squeeze(-1), o).item()
            n += 1
    return total_p / n, total_v / n


def measure_inference_time(model, dev, num_warmup=10, num_iters=100):
    model.eval()
    dummy = torch.randn(1, IN_CHANNELS, BOARD_H, BOARD_W, device=dev)
    with torch.no_grad():
        for _ in range(num_warmup):
            model(dummy)
        if dev.type == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        for _ in range(num_iters):
            model(dummy)
        if dev.type == "mps":
            torch.mps.synchronize()
    return (time.time() - t0) / num_iters * 1000


def run_experiment(name, model, train_ds, val_ds, exp_idx, total_exps):
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"[{exp_idx}/{total_exps}] {name} ({params:,} params)")
    print(f"{'='*60}")
    dev = DEVICE
    bs = BATCH_SIZE
    model = model.to(dev)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    num_batches = len(train_loader)
    history = []

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for batch_idx, (b, p, o) in enumerate(train_loader):
            b, p, o = b.to(dev), p.to(dev), o.to(dev)
            pp, pv = model(b)
            loss = -torch.sum(p * torch.log_softmax(pp, dim=1), dim=1).mean() + F.mse_loss(pv.squeeze(-1), o)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx + 1) % 200 == 0:
                print(f"    batch {batch_idx+1}/{num_batches} train_loss={running_loss/(batch_idx+1):.4f}", flush=True)
        vp, vv = evaluate(model, val_loader, dev)
        elapsed = time.time() - t0
        history.append((vp, vv))
        print(f"  epoch {epoch+1}/{EPOCHS}: val_policy={vp:.4f} val_value={vv:.4f} total={vp+vv:.4f} [{elapsed:.1f}s]", flush=True)

    infer_ms = measure_inference_time(model, dev)
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

    global EPOCHS
    if args.fast:
        max_pos = args.positions or 10_000
        EPOCHS = args.epochs if args.epochs != 5 else 2
    else:
        max_pos = args.positions
        EPOCHS = args.epochs

    print(f"Device: {DEVICE}")
    print(f"Mode: {'FAST' if args.fast else 'FULL'} | {EPOCHS} epochs | {max_pos or 'all'} positions")
    train_ds, val_ds = load_data(max_positions=max_pos)

    experiments = [
        # Baseline: best from round 2
        ("hex+SE conv-policy", HexSE_ConvPolicy(128, 6)),

        # Attention policy: 1-layer self-attn over cells, gather from/to
        ("hex+SE attn d64 1L", HexSE_AttnPolicy(128, 6, nhead=4, attn_layers=1, head_dim=64)),

        # Deeper attention
        ("hex+SE attn d64 2L", HexSE_AttnPolicy(128, 6, nhead=4, attn_layers=2, head_dim=64)),

        # Wider attention
        ("hex+SE attn d128 1L", HexSE_AttnPolicy(128, 6, nhead=8, attn_layers=1, head_dim=128)),
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

    print(f"\n{'='*80}")
    print("RESULTS (sorted by total val loss)")
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
