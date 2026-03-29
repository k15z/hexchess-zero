import { memo } from "react";
import {
  CELL_DATA,
  VIEW_BOX,
  VIEW_W,
  VIEW_H,
  HEX_SIZE,
  HEX_COLORS,
  SELECTED_COLOR,
  LAST_MOVE_OVERLAY,
  PIECE_SYMBOLS,
} from "@/engine/board-geometry";
import type { GameState } from "@/hooks/use-game";

interface Props {
  state: GameState;
  onCellClick: (q: number, r: number) => void;
}

export const HexBoard = memo(function HexBoard({ state, onCellClick }: Props) {
  const { pieces, selectedCell, legalMovesForSelected, lastMove } = state;

  const pieceMap = new Map<string, { piece: string; color: string }>();
  for (const p of pieces) {
    pieceMap.set(`${p.q},${p.r}`, p);
  }

  const legalTargets = new Set(legalMovesForSelected.map(m => `${m.to_q},${m.to_r}`));

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const target = (e.target as SVGElement).closest<SVGElement>("[data-q]");
    if (!target) return;
    const q = parseInt(target.getAttribute("data-q")!, 10);
    const r = parseInt(target.getAttribute("data-r")!, 10);
    if (!isNaN(q) && !isNaN(r)) onCellClick(q, r);
  };

  return (
    <svg
      viewBox={VIEW_BOX}
      width={VIEW_W}
      height={VIEW_H}
      className="max-w-full h-auto drop-shadow-[0_4px_16px_rgba(0,0,0,0.4)]"
      onClick={handleClick}
    >
      {/* Cells */}
      <g>
        {CELL_DATA.map(({ q, r, pts, colorIdx, key }) => {
          const isSelected = selectedCell?.q === q && selectedCell?.r === r;
          return (
            <polygon
              key={key}
              points={pts}
              fill={isSelected ? SELECTED_COLOR : HEX_COLORS[colorIdx]}
              className="hex-cell"
              data-q={q}
              data-r={r}
            />
          );
        })}
      </g>

      {/* Last move overlays */}
      {lastMove && (
        <g>
          {CELL_DATA.filter(({ q, r }) => {
            if (selectedCell?.q === q && selectedCell?.r === r) return false;
            return (
              (lastMove.from_q === q && lastMove.from_r === r) ||
              (lastMove.to_q === q && lastMove.to_r === r)
            );
          }).map(({ pts, key }) => (
            <polygon key={key} points={pts} fill={LAST_MOVE_OVERLAY} className="hex-last-move-overlay" />
          ))}
        </g>
      )}

      {/* Move indicators */}
      <g>
        {CELL_DATA.filter(({ key }) => legalTargets.has(key)).map(({ x, y, key }) => {
          if (pieceMap.has(key)) {
            return <circle key={key} cx={x} cy={y} r={HEX_SIZE * 0.82} className="move-capture-ring" />;
          }
          return <circle key={key} cx={x} cy={y} r={HEX_SIZE * 0.2} className="move-dot" />;
        })}
      </g>

      {/* Pieces */}
      <g>
        {CELL_DATA.map(({ x, y, key }) => {
          const p = pieceMap.get(key);
          if (!p) return null;
          const symbol = PIECE_SYMBOLS[p.color]?.[p.piece];
          if (!symbol) return null;
          return (
            <text key={key} x={x} y={y + 1} className={`piece-symbol piece-${p.color}`}>
              {symbol}
            </text>
          );
        })}
      </g>
    </svg>
  );
});
