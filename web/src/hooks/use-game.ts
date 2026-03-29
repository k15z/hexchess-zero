import { useCallback, useEffect, useRef, useState } from "react";
import {
  initWasm,
  createGame,
  createAiPlayer,
  createAiPlayerWithModel,
  type GameHandle,
  type AiHandle,
  type Move,
  type Piece,
} from "@/engine/wasm";

export type Side = "white" | "black";

export interface GameState {
  engineLoaded: boolean;
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
  selfPlayDelay: number;
  modelName: string | null;
  pendingPromotion: {
    fromQ: number; fromR: number; toQ: number; toR: number; moves: Move[];
  } | null;
}

const INITIAL_STATE: GameState = {
  engineLoaded: false,
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
  selfPlayDelay: 300,
  modelName: null,
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

export function useGame() {
  const gameRef = useRef<GameHandle | null>(null);
  const aiRef = useRef<AiHandle | null>(null);
  const selfPlayTimerRef = useRef<number | null>(null);
  const stateRef = useRef<GameState>(INITIAL_STATE);

  const [state, setStateRaw] = useState<GameState>(INITIAL_STATE);

  // Wrapper that also updates the ref (for use in timeouts)
  const setState = useCallback((update: Partial<GameState>) => {
    setStateRaw(prev => {
      const next = { ...prev, ...update };
      stateRef.current = next;
      return next;
    });
  }, []);

  const refresh = useCallback(() => {
    if (gameRef.current) setState(snapshot(gameRef.current));
  }, [setState]);

  // Init WASM
  useEffect(() => {
    let cancelled = false;
    initWasm().then(() => {
      if (cancelled) return;
      gameRef.current = createGame();
      aiRef.current = createAiPlayer(500);
      setState({ engineLoaded: true, ...snapshot(gameRef.current!) });
    });
    return () => {
      cancelled = true;
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
    if (!game || stateRef.current.aiThinking) return;
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
        setState({
          moveCount: stateRef.current.moveCount + 1,
          lastMove: { from_q: m.from_q, from_r: m.from_r, to_q: m.to_q, to_r: m.to_r },
          ...snapshot(game),
        });
        scheduleSelfPlayMoveRef.current();
      } catch (e) {
        console.error("Self-play error:", e);
        setState({ selfPlayMode: false });
      }
    }, stateRef.current.selfPlayDelay);
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

  const setSpeed = useCallback((val: number) => {
    stateRef.current = { ...stateRef.current, selfPlayDelay: val };
    setStateRaw(s => ({ ...s, selfPlayDelay: val }));
  }, []);

  const loadModel = useCallback(async (file: File) => {
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      aiRef.current?.free();
      aiRef.current = createAiPlayerWithModel(500, bytes);
      setState({ modelName: file.name });
      refresh();
    } catch (e) {
      console.error("Failed to load model:", e);
      alert(`Failed to load model: ${e instanceof Error ? e.message : e}`);
    }
  }, [setState, refresh]);

  const clearModel = useCallback(() => {
    aiRef.current?.free();
    aiRef.current = createAiPlayer(500);
    setState({ modelName: null });
  }, [setState]);

  return {
    state,
    actions: {
      newGame,
      selectCell,
      applyPromotion,
      cancelPromotion,
      undo,
      flipSide,
      toggleSelfPlay,
      setSpeed,
      loadModel,
      clearModel,
    },
  };
}
