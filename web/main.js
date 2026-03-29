// ---------------------------------------------------------------------------
// Hexagonal Chess – Web UI
// ---------------------------------------------------------------------------

const HEX_SIZE = 35;
const SQRT3 = Math.sqrt(3);
const BOARD_RADIUS = 5;

const PIECE_SYMBOLS = {
    white: { king: "\u2654", queen: "\u2655", rook: "\u2656", bishop: "\u2657", knight: "\u2658", pawn: "\u2659" },
    black: { king: "\u265A", queen: "\u265B", rook: "\u265C", bishop: "\u265D", knight: "\u265E", pawn: "\u265F" },
};

const HEX_COLORS = ["#f0d9b5", "#b58863", "#8b6e4f"];
const SELECTED_COLOR = "#5bc0eb";
const LAST_MOVE_OVERLAY = "rgba(170, 207, 83, 0.35)";

const STATUS_LABELS = {
    checkmate_white: "Checkmate \u2014 White wins!",
    checkmate_black: "Checkmate \u2014 Black wins!",
    stalemate: "Stalemate \u2014 Draw",
    draw_repetition: "Draw by repetition",
    draw_fifty: "Draw by fifty-move rule",
    draw_material: "Draw \u2014 insufficient material",
};

// ---- Precomputed board geometry (static — computed once) -------------------

const HEX_OFFSETS = [];
for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 180) * 60 * i;
    HEX_OFFSETS.push({ dx: HEX_SIZE * Math.cos(angle), dy: HEX_SIZE * Math.sin(angle) });
}

const CELL_DATA = [];
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
const VIEW_BOX = `${(minX - PAD).toFixed(1)} ${(minY - PAD).toFixed(1)} ${(maxX - minX + PAD * 2).toFixed(1)} ${(maxY - minY + PAD * 2).toFixed(1)}`;
const VIEW_W = Math.min(maxX - minX + PAD * 2, 720);
const VIEW_H = Math.min(maxY - minY + PAD * 2, 720);

// ---- State ----------------------------------------------------------------

let wasm = null;
let game = null;
let ai = null;
let selectedCell = null;
let legalMovesForSelected = [];
let playerColor = "white";
let aiThinking = false;
let lastMove = null;
let cachedPieceAt = {};  // populated by render(), used by click handler
let selfPlayMode = false;
let selfPlayTimer = null;
let moveCount = 0;

// ---- Helpers --------------------------------------------------------------

function clearSelection(clearLast = false) {
    selectedCell = null;
    legalMovesForSelected = [];
    if (clearLast) lastMove = null;
}

function recordLastMove(fromQ, fromR, toQ, toR) {
    lastMove = { from_q: fromQ, from_r: fromR, to_q: toQ, to_r: toR };
}

// ---- SVG helpers ----------------------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
}

// ---- Rendering ------------------------------------------------------------

function render() {
    const svg = document.getElementById("board-svg");
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    svg.setAttribute("viewBox", VIEW_BOX);
    svg.setAttribute("width", VIEW_W);
    svg.setAttribute("height", VIEW_H);

    // Board state from engine
    cachedPieceAt = {};
    if (game) {
        try {
            for (const p of game.boardState()) cachedPieceAt[`${p.q},${p.r}`] = p;
        } catch (_) { /* not ready */ }
    }

    const gCells = svgEl("g");
    const gOverlays = svgEl("g");
    const gIndicators = svgEl("g");
    const gPieces = svgEl("g");
    svg.appendChild(gCells);
    svg.appendChild(gOverlays);
    svg.appendChild(gIndicators);
    svg.appendChild(gPieces);

    const legalTargetSet = new Set(legalMovesForSelected.map(m => `${m.to_q},${m.to_r}`));

    for (const { q, r, x, y, pts, colorIdx, key } of CELL_DATA) {
        const isSelected = selectedCell && selectedCell.q === q && selectedCell.r === r;

        // Hex cell with data attributes for event delegation
        const poly = svgEl("polygon", {
            points: pts,
            fill: isSelected ? SELECTED_COLOR : HEX_COLORS[colorIdx],
            class: "hex-cell",
            "data-q": q,
            "data-r": r,
        });
        gCells.appendChild(poly);

        // Last move overlay
        if (lastMove && !isSelected) {
            const isLastFrom = lastMove.from_q === q && lastMove.from_r === r;
            const isLastTo = lastMove.to_q === q && lastMove.to_r === r;
            if (isLastFrom || isLastTo) {
                gOverlays.appendChild(svgEl("polygon", {
                    points: pts, fill: LAST_MOVE_OVERLAY, class: "hex-last-move-overlay",
                }));
            }
        }

        // Legal move indicators
        if (legalTargetSet.has(key)) {
            if (cachedPieceAt[key]) {
                gIndicators.appendChild(svgEl("circle", {
                    cx: x, cy: y, r: HEX_SIZE * 0.82, class: "move-indicator move-capture-ring",
                }));
            } else {
                gIndicators.appendChild(svgEl("circle", {
                    cx: x, cy: y, r: HEX_SIZE * 0.2, class: "move-indicator move-dot",
                }));
            }
        }

        // Piece
        const piece = cachedPieceAt[key];
        if (piece) {
            const symbol = PIECE_SYMBOLS[piece.color]?.[piece.piece];
            if (symbol) {
                const text = svgEl("text", { x, y: y + 1, class: `piece-symbol piece-${piece.color}` });
                text.textContent = symbol;
                gPieces.appendChild(text);
            }
        }
    }

    updateStatus();
}

// ---- Status bar -----------------------------------------------------------

function updateStatus() {
    const el = document.getElementById("status-text");
    el.className = "";

    if (!game) { el.textContent = "Loading engine..."; return; }

    if (aiThinking) {
        el.textContent = "AI is thinking...";
        el.classList.add("status-thinking", "thinking-pulse");
        return;
    }

    const status = game.status();
    const prefix = selfPlayMode ? `[Self-Play · Move ${moveCount}] ` : "";

    if (STATUS_LABELS[status]) {
        el.textContent = prefix + STATUS_LABELS[status];
        el.classList.add("status-gameover");
        return;
    }

    const side = game.sideToMove();
    const label = side === "white" ? "White" : "Black";
    if (game.isInCheck()) {
        el.textContent = `${prefix}${label} to move \u2014 CHECK!`;
        el.classList.add("status-check");
    } else {
        el.textContent = `${prefix}${label} to move`;
    }
}

// ---- Interaction ----------------------------------------------------------

function onCellClick(q, r) {
    if (aiThinking || selfPlayMode || !game || game.isGameOver()) return;
    if (game.sideToMove() !== playerColor) return;

    const clickedPiece = cachedPieceAt[`${q},${r}`];

    if (selectedCell) {
        const movesHere = legalMovesForSelected.filter(m => m.to_q === q && m.to_r === r);

        if (movesHere.length > 0) {
            if (movesHere.length > 1 || movesHere[0].promotion) {
                showPromotionPicker(selectedCell.q, selectedCell.r, q, r, movesHere);
                return;
            }
            applyPlayerMove(selectedCell.q, selectedCell.r, q, r, null);
            return;
        }

        if (clickedPiece && clickedPiece.color === playerColor) {
            selectPiece(q, r);
            return;
        }

        clearSelection();
        render();
        return;
    }

    if (clickedPiece && clickedPiece.color === playerColor) {
        selectPiece(q, r);
    }
}

function selectPiece(q, r) {
    selectedCell = { q, r };
    legalMovesForSelected = game.legalMoves().filter(m => m.from_q === q && m.from_r === r);
    render();
}

function applyPlayerMove(fromQ, fromR, toQ, toR, promotion) {
    game.applyMove(fromQ, fromR, toQ, toR, promotion);
    recordLastMove(fromQ, fromR, toQ, toR);
    clearSelection();
    render();

    if (!game.isGameOver() && game.sideToMove() !== playerColor) {
        scheduleAiMove();
    }
}

function scheduleAiMove() {
    aiThinking = true;
    render();

    // setTimeout lets the browser paint "thinking" status before blocking on MCTS
    setTimeout(() => {
        try {
            const m = ai.bestMove(game);
            game.applyMove(m.from_q, m.from_r, m.to_q, m.to_r, m.promotion);
            recordLastMove(m.from_q, m.from_r, m.to_q, m.to_r);
        } catch (e) {
            console.error("AI move error:", e);
        }
        aiThinking = false;
        render();
    }, 80);
}

// ---- Self-play mode -------------------------------------------------------

function startSelfPlay() {
    if (selfPlayMode) { stopSelfPlay(); return; }

    game = new wasm.Game();
    clearSelection(true);
    selfPlayMode = true;
    moveCount = 0;
    document.getElementById("btn-self-play").textContent = "Stop";
    render();
    scheduleSelfPlayMove();
}

function stopSelfPlay() {
    selfPlayMode = false;
    if (selfPlayTimer) { clearTimeout(selfPlayTimer); selfPlayTimer = null; }
    document.getElementById("btn-self-play").textContent = "Self-Play";
    render();
}

function scheduleSelfPlayMove() {
    if (!selfPlayMode || !game || game.isGameOver()) {
        if (selfPlayMode) {
            // Game ended — show result briefly, then stop
            render();
        }
        return;
    }

    const delay = parseInt(document.getElementById("speed-slider").value, 10);
    selfPlayTimer = setTimeout(() => {
        try {
            const m = ai.bestMove(game);
            game.applyMove(m.from_q, m.from_r, m.to_q, m.to_r, m.promotion);
            recordLastMove(m.from_q, m.from_r, m.to_q, m.to_r);
            moveCount++;
        } catch (e) {
            console.error("Self-play move error:", e);
            stopSelfPlay();
            return;
        }
        render();
        scheduleSelfPlayMove();
    }, delay);
}

// ---- Promotion picker -----------------------------------------------------

function showPromotionPicker(fromQ, fromR, toQ, toR, moves) {
    const overlay = document.getElementById("promotion-overlay");
    const choices = document.getElementById("promotion-choices");
    choices.innerHTML = "";

    for (const opt of ["queen", "rook", "bishop", "knight"]) {
        if (!moves.find(m => m.promotion === opt)) continue;
        const btn = document.createElement("button");
        btn.className = "promotion-btn";
        btn.textContent = PIECE_SYMBOLS[playerColor][opt];
        btn.title = opt.charAt(0).toUpperCase() + opt.slice(1);
        btn.addEventListener("click", () => {
            overlay.classList.add("hidden");
            applyPlayerMove(fromQ, fromR, toQ, toR, opt);
        });
        choices.appendChild(btn);
    }

    // Dismiss on backdrop click
    overlay.onclick = (e) => {
        if (e.target === overlay) overlay.classList.add("hidden");
    };

    overlay.classList.remove("hidden");
}

// ---- Control buttons ------------------------------------------------------

function setupControls() {
    document.getElementById("btn-new-game").addEventListener("click", () => {
        if (aiThinking) return;
        game = new wasm.Game();
        clearSelection(true);
        render();
        if (playerColor !== game.sideToMove()) scheduleAiMove();
    });

    document.getElementById("btn-undo").addEventListener("click", () => {
        if (aiThinking || !game) return;
        try { game.undoMove(); game.undoMove(); } catch (_) { /* < 2 moves */ }
        clearSelection(true);
        render();
    });

    document.getElementById("btn-flip").addEventListener("click", () => {
        if (aiThinking || selfPlayMode) return;
        playerColor = playerColor === "white" ? "black" : "white";
        clearSelection();
        if (game && !game.isGameOver() && game.sideToMove() !== playerColor) {
            scheduleAiMove();
        } else {
            render();
        }
    });

    document.getElementById("btn-self-play").addEventListener("click", () => {
        if (aiThinking) return;
        startSelfPlay();
    });

    // Event delegation: single click handler on the SVG
    document.getElementById("board-svg").addEventListener("click", (e) => {
        const cell = e.target.closest(".hex-cell");
        if (!cell) return;
        const q = parseInt(cell.getAttribute("data-q"), 10);
        const r = parseInt(cell.getAttribute("data-r"), 10);
        if (!isNaN(q) && !isNaN(r)) onCellClick(q, r);
    });
}

// ---- Init -----------------------------------------------------------------

async function init() {
    try {
        wasm = await import("../bindings/wasm/pkg/hexchess_wasm.js");
        await wasm.default();
        game = new wasm.Game();
        ai = new wasm.AiPlayer(500);
    } catch (e) {
        console.error("Failed to load WASM module:", e);
        document.getElementById("status-text").textContent =
            "Failed to load engine. Serve via HTTP (e.g. python -m http.server).";
        return;
    }

    setupControls();
    render();
}

init();
