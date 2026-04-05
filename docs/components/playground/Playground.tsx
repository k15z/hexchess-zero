"use client";

import { useCallback, useEffect, useRef, useState, memo } from "react";
import {
  initWasm,
  createGame,
  createAiPlayer,
  type GameHandle,
  type AiHandle,
  type Move,
  type Piece,
} from "./wasm";
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
} from "./hex-geometry";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Side = "white" | "black";

interface GameState {
  engineLoaded: boolean;
  loadError: string | null;
  pieces: Piece[];
  legalMovesForSelected: Move[];
  selectedCell: { q: number; r: number } | null;
  lastMove: { from_q: number; from_r: number; to_q: number; to_r: number } | null;
  status: string;
  sideToMove: string;
  isInCheck: boolean;
  isGameOver: boolean;
  playerColor: Side;
  aiThinking: boolean;
  selfPlayMode: boolean;
  moveCount: number;
  pendingPromotion: {
    fromQ: number; fromR: number; toQ: number; toR: number; moves: Move[];
  } | null;
}

const INITIAL_STATE: GameState = {
  engineLoaded: false,
  loadError: null,
  pieces: [],
  legalMovesForSelected: [],
  selectedCell: null,
  lastMove: null,
  status: "ongoing",
  sideToMove: "white",
  isInCheck: false,
  isGameOver: false,
  playerColor: "white",
  aiThinking: false,
  selfPlayMode: false,
  moveCount: 0,
  pendingPromotion: null,
};

function snapshot(game: GameHandle): Partial<GameState> {
  return {
    pieces: game.boardState(),
    status: game.status(),
    sideToMove: game.sideToMove(),
    isInCheck: game.isInCheck(),
    isGameOver: game.isGameOver(),
  };
}

// ---------------------------------------------------------------------------
// Status labels
// ---------------------------------------------------------------------------

const STATUS_LABELS: Record<string, string> = {
  checkmate_white: "Checkmate \u2014 White wins!",
  checkmate_black: "Checkmate \u2014 Black wins!",
  stalemate: "Stalemate \u2014 Draw",
  draw_repetition: "Draw by repetition",
  draw_fifty: "Draw by fifty-move rule",
  draw_material: "Draw \u2014 insufficient material",
};

// ---------------------------------------------------------------------------
// HexBoard (SVG)
// ---------------------------------------------------------------------------

const HexBoard = memo(function HexBoard({
  state,
  onCellClick,
}: {
  state: GameState;
  onCellClick: (q: number, r: number) => void;
}) {
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
    <svg viewBox={VIEW_BOX} width={VIEW_W} height={VIEW_H} className="hex-board-svg" onClick={handleClick}>
      <g>
        {CELL_DATA.map(({ q, r, pts, colorIdx, key }) => {
          const isSelected = selectedCell?.q === q && selectedCell?.r === r;
          return (
            <polygon key={key} points={pts} fill={isSelected ? SELECTED_COLOR : HEX_COLORS[colorIdx]} className="hex-cell" data-q={q} data-r={r} />
          );
        })}
      </g>
      {lastMove && (
        <g>
          {CELL_DATA.filter(({ q, r }) => {
            if (selectedCell?.q === q && selectedCell?.r === r) return false;
            return (lastMove.from_q === q && lastMove.from_r === r) || (lastMove.to_q === q && lastMove.to_r === r);
          }).map(({ pts, key }) => (
            <polygon key={key} points={pts} fill={LAST_MOVE_OVERLAY} style={{ pointerEvents: "none" }} />
          ))}
        </g>
      )}
      <g>
        {CELL_DATA.filter(({ key }) => legalTargets.has(key)).map(({ x, y, key }) => {
          if (pieceMap.has(key)) {
            return <circle key={key} cx={x} cy={y} r={HEX_SIZE * 0.82} className="move-capture-ring" />;
          }
          return <circle key={key} cx={x} cy={y} r={HEX_SIZE * 0.2} className="move-dot" />;
        })}
      </g>
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

// ---------------------------------------------------------------------------
// Promotion picker
// ---------------------------------------------------------------------------

function PromotionPicker({
  state,
  onSelect,
  onCancel,
}: {
  state: GameState;
  onSelect: (piece: string) => void;
  onCancel: () => void;
}) {
  if (!state.pendingPromotion) return null;
  const color = state.playerColor;
  const options = ["queen", "rook", "bishop", "knight"].filter(opt =>
    state.pendingPromotion?.moves.some(m => m.promotion === opt)
  );

  return (
    <div className="promotion-overlay" onClick={onCancel}>
      <div className="promotion-dialog" onClick={e => e.stopPropagation()}>
        <div className="promotion-title">Promote pawn to:</div>
        <div className="promotion-options">
          {options.map(opt => (
            <button key={opt} className="promotion-btn" onClick={() => onSelect(opt)} title={opt}>
              {PIECE_SYMBOLS[color]?.[opt]}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Playground
// ---------------------------------------------------------------------------

export function Playground() {
  const gameRef = useRef<GameHandle | null>(null);
  const aiRef = useRef<AiHandle | null>(null);
  const selfPlayTimerRef = useRef<number | null>(null);
  const stateRef = useRef<GameState>(INITIAL_STATE);
  const [state, setStateRaw] = useState<GameState>(INITIAL_STATE);

  const setState = useCallback((update: Partial<GameState> | ((prev: GameState) => Partial<GameState>)) => {
    setStateRaw(prev => {
      const partial = typeof update === "function" ? update(prev) : update;
      const next = { ...prev, ...partial };
      stateRef.current = next;
      return next;
    });
  }, []);

  // Init WASM
  useEffect(() => {
    let cancelled = false;
    initWasm()
      .then(() => {
        if (cancelled) return;
        gameRef.current = createGame();
        aiRef.current = createAiPlayer(500);
        setState({ engineLoaded: true, ...snapshot(gameRef.current!) });
      })
      .catch((err) => {
        console.error("Failed to load WASM:", err);
        if (!cancelled) {
          setState({ loadError: `Failed to load engine: ${err instanceof Error ? err.message : err}` });
        }
      });
    return () => {
      cancelled = true;
      if (selfPlayTimerRef.current) clearTimeout(selfPlayTimerRef.current);
      gameRef.current?.free();
      gameRef.current = null;
      aiRef.current?.free();
      aiRef.current = null;
    };
  }, [setState]);

  const scheduleAiMove = useCallback(() => {
    setState({ aiThinking: true });
    setTimeout(() => {
      const game = gameRef.current;
      const ai = aiRef.current;
      if (!game || !ai) return;
      try {
        const m = ai.bestMove(game);
        game.applyMove(m.from_q, m.from_r, m.to_q, m.to_r, m.promotion);
        setState({
          aiThinking: false,
          lastMove: { from_q: m.from_q, from_r: m.from_r, to_q: m.to_q, to_r: m.to_r },
          selectedCell: null,
          legalMovesForSelected: [],
          ...snapshot(game),
        });
      } catch (e) {
        console.error("AI move error:", e);
        setState({ aiThinking: false });
      }
    }, 80);
  }, [setState]);

  const newGame = useCallback(() => {
    if (selfPlayTimerRef.current) {
      clearTimeout(selfPlayTimerRef.current);
      selfPlayTimerRef.current = null;
    }
    gameRef.current?.free();
    gameRef.current = createGame();
    const s = stateRef.current;
    setState({
      selfPlayMode: false,
      selectedCell: null,
      legalMovesForSelected: [],
      lastMove: null,
      aiThinking: false,
      moveCount: 0,
      pendingPromotion: null,
      ...snapshot(gameRef.current!),
    });
    if (s.playerColor !== gameRef.current!.sideToMove()) {
      setTimeout(() => scheduleAiMove(), 0);
    }
  }, [setState, scheduleAiMove]);

  const selectCell = useCallback((q: number, r: number) => {
    const game = gameRef.current;
    const s = stateRef.current;
    if (!game || s.aiThinking || s.selfPlayMode || s.isGameOver) return;
    if (game.sideToMove() !== s.playerColor) return;

    const pieceAtClick = s.pieces.find(p => p.q === q && p.r === r);

    if (s.selectedCell) {
      const movesHere = s.legalMovesForSelected.filter(m => m.to_q === q && m.to_r === r);
      if (movesHere.length > 0) {
        if (movesHere.length > 1 || movesHere[0].promotion) {
          setState({
            pendingPromotion: {
              fromQ: s.selectedCell.q, fromR: s.selectedCell.r,
              toQ: q, toR: r, moves: movesHere,
            },
          });
          return;
        }
        const m = movesHere[0];
        game.applyMove(m.from_q, m.from_r, m.to_q, m.to_r, m.promotion);
        setState({
          selectedCell: null,
          legalMovesForSelected: [],
          lastMove: { from_q: m.from_q, from_r: m.from_r, to_q: m.to_q, to_r: m.to_r },
          pendingPromotion: null,
          ...snapshot(game),
        });
        if (!game.isGameOver() && game.sideToMove() !== s.playerColor) {
          setTimeout(() => scheduleAiMove(), 0);
        }
        return;
      }
      if (pieceAtClick && pieceAtClick.color === s.playerColor) {
        const legal = game.legalMoves().filter((m: Move) => m.from_q === q && m.from_r === r);
        setState({ selectedCell: { q, r }, legalMovesForSelected: legal });
        return;
      }
      setState({ selectedCell: null, legalMovesForSelected: [] });
      return;
    }

    if (pieceAtClick && pieceAtClick.color === s.playerColor) {
      const legal = game.legalMoves().filter((m: Move) => m.from_q === q && m.from_r === r);
      setState({ selectedCell: { q, r }, legalMovesForSelected: legal });
    }
  }, [setState, scheduleAiMove]);

  const applyPromotion = useCallback((promotion: string) => {
    const game = gameRef.current;
    const s = stateRef.current;
    if (!game || !s.pendingPromotion) return;
    const { fromQ, fromR, toQ, toR } = s.pendingPromotion;
    game.applyMove(fromQ, fromR, toQ, toR, promotion);
    setState({
      selectedCell: null,
      legalMovesForSelected: [],
      lastMove: { from_q: fromQ, from_r: fromR, to_q: toQ, to_r: toR },
      pendingPromotion: null,
      ...snapshot(game),
    });
    if (!game.isGameOver() && game.sideToMove() !== s.playerColor) {
      setTimeout(() => scheduleAiMove(), 0);
    }
  }, [setState, scheduleAiMove]);

  const cancelPromotion = useCallback(() => {
    setState({ pendingPromotion: null, selectedCell: null, legalMovesForSelected: [] });
  }, [setState]);

  const undo = useCallback(() => {
    const game = gameRef.current;
    if (!game || stateRef.current.aiThinking || stateRef.current.selfPlayMode) return;
    try { game.undoMove(); game.undoMove(); } catch { /* < 2 moves */ }
    setState({
      selectedCell: null,
      legalMovesForSelected: [],
      lastMove: null,
      ...snapshot(game),
    });
  }, [setState]);

  const flipSide = useCallback(() => {
    const s = stateRef.current;
    if (s.aiThinking || s.selfPlayMode) return;
    const next: Side = s.playerColor === "white" ? "black" : "white";
    setState({ playerColor: next, selectedCell: null, legalMovesForSelected: [] });
    if (gameRef.current && !gameRef.current.isGameOver() && gameRef.current.sideToMove() !== next) {
      setTimeout(() => scheduleAiMove(), 0);
    }
  }, [setState, scheduleAiMove]);

  // Self-play
  const scheduleSelfPlayMoveRef = useRef<() => void>(() => {});
  const scheduleSelfPlayMove = useCallback(() => {
    const s = stateRef.current;
    if (!s.selfPlayMode || !gameRef.current || gameRef.current.isGameOver()) return;
    selfPlayTimerRef.current = window.setTimeout(() => {
      const game = gameRef.current;
      const ai = aiRef.current;
      if (!game || !ai) return;
      try {
        const m = ai.bestMoveWithTemperature(game, 0.8);
        game.applyMove(m.from_q, m.from_r, m.to_q, m.to_r, m.promotion);
        setState(prev => ({
          moveCount: prev.moveCount + 1,
          lastMove: { from_q: m.from_q, from_r: m.from_r, to_q: m.to_q, to_r: m.to_r },
          ...snapshot(game),
        }));
        scheduleSelfPlayMoveRef.current();
      } catch (e) {
        console.error("Self-play error:", e);
        setState({ selfPlayMode: false });
      }
    }, 300);
  }, [setState]);
  useEffect(() => {
    scheduleSelfPlayMoveRef.current = scheduleSelfPlayMove;
  }, [scheduleSelfPlayMove]);

  const toggleSelfPlay = useCallback(() => {
    const s = stateRef.current;
    if (s.selfPlayMode) {
      if (selfPlayTimerRef.current) {
        clearTimeout(selfPlayTimerRef.current);
        selfPlayTimerRef.current = null;
      }
      setState({ selfPlayMode: false });
      return;
    }
    gameRef.current?.free();
    gameRef.current = createGame();
    setState({
      selfPlayMode: true,
      moveCount: 0,
      selectedCell: null,
      legalMovesForSelected: [],
      lastMove: null,
      ...snapshot(gameRef.current!),
    });
    setTimeout(() => scheduleSelfPlayMove(), 0);
  }, [setState, scheduleSelfPlayMove]);

  // Status text
  let statusText = "Loading engine\u2026";
  let statusClass = "";
  if (state.loadError) {
    statusText = state.loadError;
    statusClass = "status-error";
  } else if (state.engineLoaded) {
    if (state.aiThinking) {
      statusText = "AI is thinking\u2026";
      statusClass = "thinking-pulse";
    } else if (STATUS_LABELS[state.status]) {
      const prefix = state.selfPlayMode ? `[Move ${state.moveCount}] ` : "";
      statusText = `${prefix}${STATUS_LABELS[state.status]}`;
      statusClass = "status-result";
    } else {
      const side = state.sideToMove === "white" ? "White" : "Black";
      const prefix = state.selfPlayMode ? `[Move ${state.moveCount}] ` : "";
      statusText = state.isInCheck ? `${prefix}${side} to move \u2014 CHECK!` : `${prefix}${side} to move`;
      statusClass = state.isInCheck ? "status-check" : "";
    }
  }

  return (
    <div className="playground-container">
      <div className="playground-status">
        <span className={statusClass}>{statusText}</span>
      </div>

      <HexBoard state={state} onCellClick={selectCell} />

      <div className="playground-controls">
        <button className="pg-btn" onClick={newGame} disabled={state.aiThinking || !state.engineLoaded}>
          New Game
        </button>
        <button className="pg-btn" onClick={undo} disabled={state.aiThinking || state.selfPlayMode || !state.engineLoaded}>
          Undo
        </button>
        <button className="pg-btn" onClick={flipSide} disabled={state.aiThinking || state.selfPlayMode || !state.engineLoaded}>
          Flip Side
        </button>
        <button
          className={`pg-btn ${state.selfPlayMode ? "pg-btn-active" : ""}`}
          onClick={toggleSelfPlay}
          disabled={state.aiThinking || !state.engineLoaded}
        >
          {state.selfPlayMode ? "Stop" : "AI vs AI"}
        </button>
      </div>

      <PromotionPicker state={state} onSelect={applyPromotion} onCancel={cancelPromotion} />
    </div>
  );
}
