// WASM singleton loader for the docs playground

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let wasm: any = null;

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

export async function initWasm() {
  if (wasm) return;
  // Build URL dynamically so webpack/turbopack won't try to resolve it.
  const base = window.location.origin;
  const jsUrl = `${base}/wasm/hexchess_wasm.js`;
  const wasmUrl = `${base}/wasm/hexchess_wasm_bg.wasm`;
  wasm = await (Function(`return import("${jsUrl}")`)() as Promise<any>);
  await wasm.default(wasmUrl);
}

export function createGame(): GameHandle {
  return new wasm.Game();
}

export function createAiPlayer(sims: number): AiHandle {
  return new wasm.AiPlayer(sims);
}
