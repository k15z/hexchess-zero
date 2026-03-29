// WASM singleton loader

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let wasm: any = null;

export async function initWasm() {
  if (wasm) return;
  // @ts-expect-error -- WASM module resolved at runtime via Vite alias
  wasm = await import("hexchess-wasm/hexchess_wasm.js");
  await wasm.default();
}

export interface Piece {
  q: number;
  r: number;
  piece: string;
  color: "white" | "black";
}

export interface Move {
  from_q: number;
  from_r: number;
  to_q: number;
  to_r: number;
  promotion: string | null;
}

export interface GameHandle {
  legalMoves(): Move[];
  boardState(): Piece[];
  applyMove(fq: number, fr: number, tq: number, tr: number, promo: string | null): void;
  undoMove(): void;
  status(): string;
  isGameOver(): boolean;
  sideToMove(): string;
  isInCheck(): boolean;
  free(): void;
}

export interface AiHandle {
  bestMove(game: GameHandle): Move;
  bestMoveWithTemperature(game: GameHandle, temp: number): Move;
  free(): void;
}

export function createGame(): GameHandle {
  return new wasm.Game();
}

export function createAiPlayer(sims: number): AiHandle {
  return new wasm.AiPlayer(sims);
}

export function createAiPlayerWithModel(sims: number, modelBytes: Uint8Array): AiHandle {
  return wasm.AiPlayer.withModel(sims, modelBytes);
}
