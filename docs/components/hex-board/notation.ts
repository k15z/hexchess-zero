// Glinski hexagonal-chess notation.
//
// Files: a b c d e f g h i k l (j is skipped — 11 files total).
// Ranks: 1..11. Center column is `f`. Each file has a different range of
// ranks because of the hex shape (file f has 11 cells, files a/l have 6).
//
// Mapping from axial (q, r):
//   file = ['a','b','c','d','e','f','g','h','i','k','l'][q + 5]
//   rank = r + 6 + min(q, 0)
//
// This must stay in sync with `HexCoord::to_notation` in engine/src/board.rs.

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l"] as const;

export function toNotation(q: number, r: number): string {
  const file = FILES[q + 5];
  const rank = r + 6 + Math.min(q, 0);
  return `${file}${rank}`;
}

export function fromNotation(s: string): [number, number] | null {
  if (s.length < 2) return null;
  const fileIdx = FILES.indexOf(s[0] as (typeof FILES)[number]);
  if (fileIdx < 0) return null;
  const rank = parseInt(s.slice(1), 10);
  if (!Number.isFinite(rank) || rank < 1 || rank > 11) return null;
  const q = fileIdx - 5;
  const r = rank - 6 - Math.min(q, 0);
  if (r < -5 || r > 5) return null;
  if (Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r)) > 5) return null;
  return [q, r];
}
