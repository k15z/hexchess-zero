# 05 — Network architecture

## The basic shape

```
Input planes (C_in × H × W)
   ↓
Initial conv (C filters, 3×3) + BN + ReLU
   ↓
N residual blocks (each: 2× [conv 3×3 + BN + ReLU], plus skip)
   ↓
   ├── Policy head: 1×1 conv → flatten → FC → softmax over moves
   └── Value head:  1×1 conv → flatten → FC → tanh (or → 3-class softmax for WDL)
```

## Sizing the trunk

Reference points:

| System | Blocks | Filters | Params | Notes |
|---|---|---|---|---|
| AlphaGo Zero (small) | 20 | 256 | ~22M | |
| AlphaZero (chess/shogi) | 19 | 256 | ~22M | |
| AlphaZero (Go) | 39 | 256 | ~46M | |
| KataGo final main run | 20 | 256 | ~24M | + SE blocks |
| KataGo b40 | 40 | 256 | ~50M | |
| Lc0 T40 | 20 | 256 | ~24M | early production |
| Lc0 T60 | 24 | 320 | ~46M | |
| Lc0 BT4 / 40b | 40 | 512 | ~200M+ | current top-tier |

For a small board game (~91 cells, branching ~30–50), **start with 6–10 blocks × 96–192
filters** (~1–4M params). You can scale up later. KataGo's progressive growing went
`(6,96) → (10,128) → (15,192) → (20,256)` over the course of one run.

Rules of thumb:
- Going deeper helps more than going wider (up to ~20 blocks).
- Doubling filters quadruples FLOPs; doubling blocks doubles FLOPs.
- A network that's too small caps your eventual Elo. Don't undersize — but also don't
  oversize early when you don't have data to fill it.

## Squeeze-and-Excitation blocks

After the second conv in each residual block, add an SE branch
([Hu et al. 2018](https://arxiv.org/abs/1709.01507)):

```
y = conv2(relu(conv1(x)))             # main branch
g = global_avg_pool(y)                # (C,)
s = sigmoid(W2 · relu(W1 · g))        # (C,) gating, W1: C→C/r, W2: C/r→C
out = relu(x + y * s.reshape(C,1,1))
```

Reduction ratio `r = 4` or `8`. Adds <5% params, gives consistent ~50–100 Elo on board game nets
([Cazenave 2021](https://arxiv.org/abs/2102.03467), KataGo follow-ups).

## Global pooling layers

Even more important than SE for board games: at a few residual blocks (e.g. depths 1/3 and 2/3
through the trunk), include a "global pooling bias" branch that mixes mean+max across the spatial
dimensions and adds a per-channel bias to the spatial features. KataGo attributes **~1.6x training
speedup** to this alone.

Without it, a 20-block 3x3-conv stack can only see ~40 cells of context, which is a problem on
any board larger than ~7x7.

## Value head: WDL vs scalar

**Use WDL.** Three-class softmax (Win / Draw / Loss), trained with cross-entropy against the
empirical game outcome (one-hot for the actual result).

- Strictly more expressive than a `tanh` scalar.
- Calibrates draw probabilities cleanly.
- Required for any sensible "contempt" or "drawishness" reasoning.
- At inference, scalar `v = P(W) − P(L)` recovers the AlphaZero scalar but with strictly more
  information available downstream.

Lc0 ([blog](https://lczero.org/blog/2020/04/wdl-head/)) found WDL better than scalar across all
phases of training.

For games with no draws, scalar is fine — but most "interesting" board games have draws.

## Policy head

Original AlphaZero: `1×1 conv → flatten → FC → softmax`. The number of output indices is fixed
across positions; illegal moves are masked out (set to `−∞` before softmax).

Move encoding for board games typically uses **(from-square, to-square)** pairs, with extra
indices for promotions/special moves. For a 91-cell hex board, pure `91 × 91 = 8281` is wasteful;
prune to legal piece-direction combos to get ~4000 indices.

The crucial property: **the move-index mapping must be deterministic and stable**. If you ever
change it, all your old training data becomes invalid.

## Auxiliary heads

Beyond `(p, v)`, consider:

- **Moves-left head** (Lc0): scalar prediction of plies-to-end. Helps the engine make progress
  in winning positions.
- **Ownership / score** (KataGo): only relevant for territory games.
- **Auxiliary policy** (KataGo): policy after opponent's reply. Helps with forcing sequences.
- **Short-term value** (KataGo): exponential averages at horizons of ~6, ~16, ~50 plies. Lower
  variance than terminal-only `z`, useful for early training.
- **In-check / threats** (chess-like): cheap to compute as labels, dense regularization.

Each aux head costs a few % FLOPs and a few % params. Net effect on training efficiency is
positive whenever the auxiliary signal is correlated with the main task.

## Input planes

Channels in the input tensor. Typical for a chess-like game:

| # | Channels | Content |
|---|---|---|
| 12 | piece type × color | one-hot per cell |
| 1 | side-to-move | constant plane |
| 1 | castling rights | (or other special-move state) |
| 1 | en-passant target | |
| 1 | halfmove clock | normalized |
| 1 | fullmove number | normalized |
| 1 | repetition count | for threefold detection |
| 1 | validity / boundary mask | only matters for non-rectangular boards |
| 1 | in-check indicator | optional but cheap |

KataGo for Go uses ~22 input features including liberties, last-N moves, ko state, etc. Add
features that are *cheap to compute* and *capture information the network would otherwise need
many layers to derive*. The "in-check" plane in chess is the canonical example.

For non-rectangular boards (hex grids), include a **validity mask** plane and zero out invalid
cells in all other planes. Pad the hex grid into the smallest enclosing rectangle.

## Activations and normalization

- **Activation**: ReLU is standard. Mish/Swish provides marginal gains in some Go nets
  ([Cazenave](https://www.lamsade.dauphine.fr/~cazenave/papers/CosineAnnealingMixnetAndSwishActivationForComputerGo.pdf)).
- **Normalization**: BatchNorm is standard. KataGo eventually moved to **Fixup/no-BN** with
  variance-preserving initialization ([KataGoMethods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md))
  to avoid train/eval discrepancies. If you stay with BN, **make sure you're using `eval()`
  mode at inference** — this is one of the most common bugs (see `10-common-failure-modes.md`).

## A reasonable starter for a 91-cell hex chess engine

- 10 residual blocks × 192 filters
- SE in every block (reduction 8)
- Global-pool-bias branch in blocks 3 and 6
- Input: ~19 planes, embedded in 11×11 with validity mask
- Policy head: 1×1 conv → FC → ~4000 outputs (move-index bijection)
- Value head: 1×1 conv → FC → 3 logits (WDL)
- (Optional) Moves-left head: scalar regression
- ~3M params, fits comfortably on a single consumer GPU

This is roughly what KataGo started its progressive run with, and matches the current
hex-chess project's net.
