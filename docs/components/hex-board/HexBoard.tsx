"use client";

import { memo } from "react";
import {
  CELL_DATA,
  VIEW_BOX,
  HEX_SIZE,
  HEX_COLORS,
  PIECE_SYMBOLS,
  type CellData,
} from "../playground/hex-geometry";
import "./hex-board.css";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BoardPiece {
  q: number;
  r: number;
  piece: "king" | "queen" | "rook" | "bishop" | "knight" | "pawn";
  color: "white" | "black";
}

export interface CellHighlight {
  q: number;
  r: number;
  color?: string;
}

export interface Arrow {
  from: [number, number];
  to: [number, number];
  color?: string;
}

export interface HexBoardProps {
  pieces?: BoardPiece[];
  highlights?: CellHighlight[];
  arrows?: Arrow[];
  showCoordinates?: boolean;
  size?: "sm" | "md" | "lg";
  caption?: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SIZE_PX = { sm: 320, md: 480, lg: 600 } as const;

const DEFAULT_HIGHLIGHT = "rgba(90, 192, 235, 0.45)";

const HIGHLIGHT_RED = "rgba(231, 76, 60, 0.3)";
const HIGHLIGHT_GREEN = "rgba(46, 204, 113, 0.35)";
const HIGHLIGHT_PURPLE = "rgba(155, 89, 182, 0.4)";
const HIGHLIGHT_YELLOW = "rgba(241, 196, 15, 0.4)";
const HIGHLIGHT_GREEN_STRONG = "rgba(46, 204, 113, 0.45)";
const HIGHLIGHT_RED_STRONG = "rgba(231, 76, 60, 0.35)";

const ARROW_RED = "#e74c3c";
const ARROW_GREEN = "#2ecc71";

const CELL_MAP = new Map<string, CellData>();
for (const c of CELL_DATA) CELL_MAP.set(c.key, c);

function highlight(cells: [number, number][], color: string): CellHighlight[] {
  return cells.map(([q, r]) => ({ q, r, color }));
}

// ---------------------------------------------------------------------------
// Arrow rendering
// ---------------------------------------------------------------------------

function ArrowLine({ arrow, idx }: { arrow: Arrow; idx: number }) {
  const from = CELL_MAP.get(`${arrow.from[0]},${arrow.from[1]}`);
  const to = CELL_MAP.get(`${arrow.to[0]},${arrow.to[1]}`);
  if (!from || !to) return null;

  const color = arrow.color ?? ARROW_RED;
  const markerId = `arrow-${idx}`;

  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  const shorten = HEX_SIZE * 0.45;
  const toX = to.x - (dx / len) * shorten;
  const toY = to.y - (dy / len) * shorten;
  const fromX = from.x + (dx / len) * shorten * 0.5;
  const fromY = from.y + (dy / len) * shorten * 0.5;

  return (
    <>
      <marker
        id={markerId}
        viewBox="0 0 10 10"
        refX="8"
        refY="5"
        markerWidth="5"
        markerHeight="5"
        orient="auto-start-reverse"
      >
        <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
      </marker>
      <line
        x1={fromX}
        y1={fromY}
        x2={toX}
        y2={toY}
        stroke={color}
        strokeWidth={3.5}
        markerEnd={`url(#${markerId})`}
        opacity={0.85}
      />
    </>
  );
}

// ---------------------------------------------------------------------------
// HexBoard
// ---------------------------------------------------------------------------

export const HexBoard = memo(function HexBoard({
  pieces = [],
  highlights = [],
  arrows = [],
  showCoordinates = false,
  size = "md",
  caption,
}: HexBoardProps) {
  const width = SIZE_PX[size];

  const pieceMap = new Map<string, BoardPiece>();
  for (const p of pieces) pieceMap.set(`${p.q},${p.r}`, p);

  const highlightMap = new Map<string, string>();
  for (const h of highlights) highlightMap.set(`${h.q},${h.r}`, h.color ?? DEFAULT_HIGHLIGHT);

  return (
    <figure className="diagram-figure">
      <svg
        viewBox={VIEW_BOX}
        width={width}
        className="hex-board-static"
      >
        {/* Cell polygons */}
        <g>
          {CELL_DATA.map(({ pts, colorIdx, key }) => (
            <polygon
              key={key}
              points={pts}
              fill={HEX_COLORS[colorIdx]}
              className="hex-board-cell"
            />
          ))}
        </g>

        {/* Highlight overlays */}
        <g>
          {CELL_DATA.map(({ key, pts }) => {
            const color = highlightMap.get(key);
            if (!color) return null;
            return (
              <polygon
                key={`hl-${key}`}
                points={pts}
                fill={color}
                style={{ pointerEvents: "none" }}
              />
            );
          })}
        </g>

        {/* Coordinate labels */}
        {showCoordinates && (
          <g>
            {CELL_DATA.map(({ q, r, x, y, key }) => {
              const hasPiece = pieceMap.has(key);
              return (
                <text
                  key={`coord-${key}`}
                  x={x}
                  y={hasPiece ? y + HEX_SIZE * 0.52 : y + 1}
                  className="hex-board-coord"
                >
                  {q},{r}
                </text>
              );
            })}
          </g>
        )}

        {/* Pieces */}
        <g>
          {CELL_DATA.map(({ x, y, key }) => {
            const p = pieceMap.get(key);
            if (!p) return null;
            const symbol = PIECE_SYMBOLS[p.color]?.[p.piece];
            if (!symbol) return null;
            const yOffset = showCoordinates ? -HEX_SIZE * 0.15 : 1;
            return (
              <text
                key={`piece-${key}`}
                x={x}
                y={y + yOffset}
                className={`hex-board-cell-piece hex-board-piece-${p.color}`}
              >
                {symbol}
              </text>
            );
          })}
        </g>

        {/* Arrows (markers defined inline per arrow) */}
        <g>
          {arrows.map((arrow, idx) => (
            <ArrowLine key={idx} arrow={arrow} idx={idx} />
          ))}
        </g>
      </svg>

      {caption && <figcaption className="diagram-caption">{caption}</figcaption>}
    </figure>
  );
});

// ---------------------------------------------------------------------------
// Preset data
// ---------------------------------------------------------------------------

const GLINSKI_START: BoardPiece[] = [
  { q: -4, r: -1, piece: "pawn", color: "white" },
  { q: -3, r: -1, piece: "pawn", color: "white" },
  { q: -2, r: -1, piece: "pawn", color: "white" },
  { q: -1, r: -1, piece: "pawn", color: "white" },
  { q: 0, r: -1, piece: "pawn", color: "white" },
  { q: 1, r: -2, piece: "pawn", color: "white" },
  { q: 2, r: -3, piece: "pawn", color: "white" },
  { q: 3, r: -4, piece: "pawn", color: "white" },
  { q: 4, r: -5, piece: "pawn", color: "white" },
  { q: -3, r: -2, piece: "rook", color: "white" },
  { q: 3, r: -5, piece: "rook", color: "white" },
  { q: -2, r: -3, piece: "knight", color: "white" },
  { q: 2, r: -5, piece: "knight", color: "white" },
  { q: 0, r: -5, piece: "bishop", color: "white" },
  { q: 0, r: -4, piece: "bishop", color: "white" },
  { q: 0, r: -3, piece: "bishop", color: "white" },
  { q: -1, r: -4, piece: "queen", color: "white" },
  { q: 1, r: -5, piece: "king", color: "white" },
  { q: 4, r: 1, piece: "pawn", color: "black" },
  { q: 3, r: 1, piece: "pawn", color: "black" },
  { q: 2, r: 1, piece: "pawn", color: "black" },
  { q: 1, r: 1, piece: "pawn", color: "black" },
  { q: 0, r: 1, piece: "pawn", color: "black" },
  { q: -1, r: 2, piece: "pawn", color: "black" },
  { q: -2, r: 3, piece: "pawn", color: "black" },
  { q: -3, r: 4, piece: "pawn", color: "black" },
  { q: -4, r: 5, piece: "pawn", color: "black" },
  { q: 3, r: 2, piece: "rook", color: "black" },
  { q: -3, r: 5, piece: "rook", color: "black" },
  { q: 2, r: 3, piece: "knight", color: "black" },
  { q: -2, r: 5, piece: "knight", color: "black" },
  { q: 0, r: 5, piece: "bishop", color: "black" },
  { q: 0, r: 4, piece: "bishop", color: "black" },
  { q: 0, r: 3, piece: "bishop", color: "black" },
  { q: 1, r: 4, piece: "queen", color: "black" },
  { q: -1, r: 5, piece: "king", color: "black" },
];

// Precomputed highlight arrays (hoisted to avoid per-render allocation)

const PROMOTION_HIGHLIGHTS: CellHighlight[] = [
  ...highlight([[-5,5],[-4,5],[-3,5],[-2,5],[-1,5],[0,5],[1,4],[2,3],[3,2],[4,1],[5,0]], HIGHLIGHT_GREEN_STRONG),
  ...highlight([[5,-5],[4,-5],[3,-5],[2,-5],[1,-5],[0,-5],[-1,-4],[-2,-3],[-3,-2],[-4,-1],[-5,0]], HIGHLIGHT_RED_STRONG),
];

const ROOK_HIGHLIGHTS: CellHighlight[] = highlight([
  [1,0],[2,0],[3,0],[4,0],[5,0],[-1,0],[-2,0],[-3,0],[-4,0],[-5,0],
  [0,1],[0,2],[0,3],[0,4],[0,5],[0,-1],[0,-2],[0,-3],[0,-4],[0,-5],
  [1,-1],[2,-2],[3,-3],[4,-4],[5,-5],[-1,1],[-2,2],[-3,3],[-4,4],[-5,5],
], HIGHLIGHT_RED);

const BISHOP_HIGHLIGHTS: CellHighlight[] = highlight([
  [2,-1],[4,-2],[-2,1],[-4,2],
  [1,1],[2,2],[3,2],[-1,-1],[-2,-2],[-3,-2],
  [1,-2],[2,-4],[3,-5],[-1,2],[-2,4],[-3,5],
], HIGHLIGHT_GREEN);

const KNIGHT_HIGHLIGHTS: CellHighlight[] = highlight([
  [1,2],[2,1],[3,-1],[1,-3],[-1,-2],[-2,-1],
  [-3,1],[-1,3],[-2,3],[-3,2],[2,-3],[3,-2],
], HIGHLIGHT_PURPLE);

const KING_HIGHLIGHTS: CellHighlight[] = highlight([
  [1,0],[-1,0],[0,1],[0,-1],[1,-1],[-1,1],
  [2,-1],[-2,1],[1,1],[-1,-1],[1,-2],[-1,2],
], HIGHLIGHT_YELLOW);

// Hoisted piece/arrow arrays so preset components pass stable references to memo'd HexBoard

const CENTER_ROOK: BoardPiece[] = [{ q: 0, r: 0, piece: "rook", color: "white" }];
const CENTER_BISHOP: BoardPiece[] = [{ q: 0, r: 0, piece: "bishop", color: "white" }];
const CENTER_KNIGHT: BoardPiece[] = [{ q: 0, r: 0, piece: "knight", color: "white" }];
const CENTER_KING: BoardPiece[] = [{ q: 0, r: 0, piece: "king", color: "white" }];
const CENTER_PAWN: BoardPiece[] = [{ q: 0, r: 0, piece: "pawn", color: "white" }];
const CENTER_QUEEN: BoardPiece[] = [{ q: 0, r: 0, piece: "queen", color: "white" }];

const PAWN_ARROWS: Arrow[] = [
  { from: [0, 0], to: [0, 1], color: ARROW_GREEN },
  { from: [0, 0], to: [0, 2], color: ARROW_GREEN },
  { from: [0, 0], to: [-1, 1], color: ARROW_RED },
  { from: [0, 0], to: [1, 0], color: ARROW_RED },
];

const DIRECTION_ARROWS: Arrow[] = [
  { from: [0, 0], to: [2, 0], color: ARROW_RED },
  { from: [0, 0], to: [-2, 0], color: ARROW_RED },
  { from: [0, 0], to: [0, 2], color: ARROW_RED },
  { from: [0, 0], to: [0, -2], color: ARROW_RED },
  { from: [0, 0], to: [2, -2], color: ARROW_RED },
  { from: [0, 0], to: [-2, 2], color: ARROW_RED },
  { from: [0, 0], to: [3, -1], color: ARROW_GREEN },
  { from: [0, 0], to: [-3, 1], color: ARROW_GREEN },
  { from: [0, 0], to: [2, 2], color: ARROW_GREEN },
  { from: [0, 0], to: [-2, -2], color: ARROW_GREEN },
  { from: [0, 0], to: [2, -4], color: ARROW_GREEN },
  { from: [0, 0], to: [-2, 4], color: ARROW_GREEN },
];

// ---------------------------------------------------------------------------
// Preset components
// ---------------------------------------------------------------------------

export function StartingPosition() {
  return (
    <HexBoard
      pieces={GLINSKI_START}
      size="lg"
      caption="The Glinski starting position. Note 3 bishops stacked along the center file and 9 pawns per side."
    />
  );
}

export function PromotionZones() {
  return (
    <HexBoard
      highlights={PROMOTION_HIGHLIGHTS}
      size="md"
      caption="Promotion zones: green = white promotes here, red = black promotes here"
    />
  );
}

export function CoordinateBoard() {
  return (
    <HexBoard
      showCoordinates
      size="lg"
      caption="The 91-cell hex board with axial coordinates (q, r). The center cell is (0, 0)."
    />
  );
}

export function RookMovement() {
  return (
    <HexBoard
      pieces={CENTER_ROOK}
      highlights={ROOK_HIGHLIGHTS}
      size="md"
      caption="Rook movement: 6 cardinal directions through cell edges"
    />
  );
}

export function BishopMovement() {
  return (
    <HexBoard
      pieces={CENTER_BISHOP}
      highlights={BISHOP_HIGHLIGHTS}
      size="md"
      caption="Bishop movement: 6 diagonal directions through cell vertices. Bishops stay on one cell color."
    />
  );
}

export function KnightMovement() {
  return (
    <HexBoard
      pieces={CENTER_KNIGHT}
      highlights={KNIGHT_HIGHLIGHTS}
      size="md"
      caption="Knight movement: 12 possible jump destinations from center (vs 8 in standard chess)"
    />
  );
}

export function KingMovement() {
  return (
    <HexBoard
      pieces={CENTER_KING}
      highlights={KING_HIGHLIGHTS}
      size="sm"
      caption="King movement: one step in any of 12 directions"
    />
  );
}

export function PawnMovement() {
  return (
    <HexBoard
      pieces={CENTER_PAWN}
      arrows={PAWN_ARROWS}
      size="sm"
      caption="White pawn: advances in +r (green), captures diagonally (red). Double advance from starting position only."
    />
  );
}

export function DirectionSystem() {
  return (
    <HexBoard
      pieces={CENTER_QUEEN}
      arrows={DIRECTION_ARROWS}
      size="md"
      caption="Red = 6 cardinal directions (rook), Green = 6 diagonal directions (bishop). The queen uses all 12."
    />
  );
}
