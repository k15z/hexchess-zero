export const HEX_SIZE = 35;
const SQRT3 = Math.sqrt(3);
const BOARD_RADIUS = 5;

export const PIECE_SYMBOLS: Record<string, Record<string, string>> = {
  white: { king: "\u2654", queen: "\u2655", rook: "\u2656", bishop: "\u2657", knight: "\u2658", pawn: "\u2659" },
  black: { king: "\u265A", queen: "\u265B", rook: "\u265C", bishop: "\u265D", knight: "\u265E", pawn: "\u265F" },
};

export const HEX_COLORS = ["#f0d9b5", "#b58863", "#8b6e4f"];
export const SELECTED_COLOR = "#5bc0eb";
export const LAST_MOVE_OVERLAY = "rgba(170, 207, 83, 0.35)";

const HEX_OFFSETS: { dx: number; dy: number }[] = [];
for (let i = 0; i < 6; i++) {
  const angle = (Math.PI / 180) * 60 * i;
  HEX_OFFSETS.push({ dx: HEX_SIZE * Math.cos(angle), dy: HEX_SIZE * Math.sin(angle) });
}

export interface CellData {
  q: number;
  r: number;
  x: number;
  y: number;
  pts: string;
  colorIdx: number;
  key: string;
}

export const CELL_DATA: CellData[] = [];

let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
for (let q = -BOARD_RADIUS; q <= BOARD_RADIUS; q++) {
  for (let r = -BOARD_RADIUS; r <= BOARD_RADIUS; r++) {
    if (Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r)) <= BOARD_RADIUS) {
      const x = HEX_SIZE * (3 / 2) * q;
      const y = HEX_SIZE * ((SQRT3 / 2) * q + SQRT3 * r);
      const pts = HEX_OFFSETS.map(o => `${(x + o.dx).toFixed(2)},${(y + o.dy).toFixed(2)}`).join(" ");
      const colorIdx = ((q - r) % 3 + 3) % 3;
      CELL_DATA.push({ q, r, x, y, pts, colorIdx, key: `${q},${r}` });
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
  }
}

const PAD = HEX_SIZE + 4;
export const VIEW_BOX = `${(minX - PAD).toFixed(1)} ${(minY - PAD).toFixed(1)} ${(maxX - minX + PAD * 2).toFixed(1)} ${(maxY - minY + PAD * 2).toFixed(1)}`;
export const VIEW_W = Math.min(maxX - minX + PAD * 2, 720);
export const VIEW_H = Math.min(maxY - minY + PAD * 2, 720);
