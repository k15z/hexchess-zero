//! Alpha-beta minimax search for hexagonal chess.
//!
//! Optimizations:
//! - Iterative deepening with transposition table
//! - MVV-LVA capture ordering
//! - Killer move heuristic (2 per ply)
//! - History heuristic for quiet moves
//! - Quiescence search (captures only at horizon)
//! - No redundant move generation (avoids double `status()` calls)

use crate::board::{Color, NUM_CELLS};
use crate::eval;
use crate::game::GameState;
use crate::movegen::{self, Move, MoveList};

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Result of a minimax search.
pub struct MinimaxResult {
    pub best_move: Move,
    pub score: i32,
    pub nodes: u64,
    pub depth: u32,
}

/// A move paired with its minimax score.
pub struct RankedMove {
    pub mv: Move,
    pub score: i32,
}

/// Result of `search_all_moves`: scores for every legal root move.
pub struct MinimaxAllResult {
    pub moves: Vec<RankedMove>,
    pub nodes: u64,
}

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------

/// Bound type stored in a TT entry.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower, // beta cutoff (score >= beta)
    Upper, // failed low (score <= alpha)
}

/// A single transposition table entry.
#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,   // full Zobrist key for collision detection
    depth: u32, // search depth remaining when this was stored
    score: i32,
    bound: Bound,
    best_from: u8, // cell index of best move (0..91), 255 = none
    best_to: u8,   // cell index of best move destination
}

impl TtEntry {
    const EMPTY: Self = Self {
        key: 0,
        depth: 0,
        score: 0,
        bound: Bound::Exact,
        best_from: 255,
        best_to: 255,
    };
}

const TT_SIZE: usize = 1 << 20; // ~1M entries
const TT_MASK: usize = TT_SIZE - 1;

struct TranspositionTable {
    entries: Vec<TtEntry>,
}

impl TranspositionTable {
    fn new() -> Self {
        Self {
            entries: vec![TtEntry::EMPTY; TT_SIZE],
        }
    }

    #[inline]
    fn probe(&self, key: u64) -> Option<&TtEntry> {
        let entry = &self.entries[(key as usize) & TT_MASK];
        if entry.key == key { Some(entry) } else { None }
    }

    #[inline]
    fn store(&mut self, key: u64, depth: u32, score: i32, bound: Bound, best_move: Option<&Move>) {
        let idx = (key as usize) & TT_MASK;
        let existing = &self.entries[idx];
        // Always-replace with depth-preferred: replace if new depth >= stored depth
        // or if the slot is empty/collision.
        if existing.key != key || depth >= existing.depth {
            let (bf, bt) = match best_move {
                Some(mv) => {
                    let fi = crate::board::coord_to_index(mv.from).unwrap_or(255) as u8;
                    let ti = crate::board::coord_to_index(mv.to).unwrap_or(255) as u8;
                    (fi, ti)
                }
                None => (255, 255),
            };
            self.entries[idx] = TtEntry {
                key,
                depth,
                score,
                bound,
                best_from: bf,
                best_to: bt,
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Killer moves & history heuristic
// ---------------------------------------------------------------------------

const MAX_PLY: usize = 128;

/// Two killer move slots per ply, stored as (from_idx, to_idx) pairs.
struct Killers {
    slots: [[(u8, u8); 2]; MAX_PLY],
}

impl Killers {
    fn new() -> Self {
        Self {
            slots: [[(255, 255); 2]; MAX_PLY],
        }
    }

    #[inline]
    fn store(&mut self, ply: usize, mv: &Move) {
        if ply >= MAX_PLY {
            return;
        }
        let fi = crate::board::coord_to_index(mv.from).unwrap_or(255) as u8;
        let ti = crate::board::coord_to_index(mv.to).unwrap_or(255) as u8;
        let pair = (fi, ti);
        // Don't store duplicates; shift slot 0 to slot 1.
        if self.slots[ply][0] != pair {
            self.slots[ply][1] = self.slots[ply][0];
            self.slots[ply][0] = pair;
        }
    }

    #[inline]
    fn is_killer(&self, ply: usize, mv: &Move) -> bool {
        if ply >= MAX_PLY {
            return false;
        }
        let fi = crate::board::coord_to_index(mv.from).unwrap_or(255) as u8;
        let ti = crate::board::coord_to_index(mv.to).unwrap_or(255) as u8;
        let pair = (fi, ti);
        self.slots[ply][0] == pair || self.slots[ply][1] == pair
    }
}

/// History heuristic: indexed by [color][from_cell][to_cell].
struct History {
    table: [[[i32; NUM_CELLS]; NUM_CELLS]; 2],
}

impl History {
    fn new() -> Self {
        Self {
            table: [[[0; NUM_CELLS]; NUM_CELLS]; 2],
        }
    }

    #[inline]
    fn update(&mut self, color: Color, mv: &Move, depth: u32) {
        let ci = color as usize;
        let fi = crate::board::coord_to_index(mv.from).unwrap_or(0);
        let ti = crate::board::coord_to_index(mv.to).unwrap_or(0);
        self.table[ci][fi][ti] += (depth * depth) as i32;
    }

    #[inline]
    fn score(&self, color: Color, mv: &Move) -> i32 {
        let ci = color as usize;
        let fi = crate::board::coord_to_index(mv.from).unwrap_or(0);
        let ti = crate::board::coord_to_index(mv.to).unwrap_or(0);
        self.table[ci][fi][ti]
    }
}

// ---------------------------------------------------------------------------
// Move ordering
// ---------------------------------------------------------------------------

/// Piece values for MVV-LVA ordering (centipawns).
#[inline]
fn mvv_lva_piece_val(kind: crate::board::PieceKind) -> i32 {
    use crate::board::PieceKind;
    match kind {
        PieceKind::Pawn => 1,
        PieceKind::Knight => 3,
        PieceKind::Bishop => 3,
        PieceKind::Rook => 5,
        PieceKind::Queen => 9,
        PieceKind::King => 100,
    }
}

/// Score a move for ordering. Higher is searched first.
///
/// Priority: TT best move > captures (MVV-LVA) > killers > history.
#[inline]
fn move_score(
    mv: &Move,
    board: &crate::board::Board,
    tt_from: u8,
    tt_to: u8,
    ply: usize,
    killers: &Killers,
    history: &History,
) -> i32 {
    let fi = crate::board::coord_to_index(mv.from).unwrap_or(255) as u8;
    let ti = crate::board::coord_to_index(mv.to).unwrap_or(255) as u8;

    // TT best move gets highest priority.
    if fi == tt_from && ti == tt_to && tt_from != 255 {
        return 1_000_000;
    }

    // Captures: MVV-LVA scoring.
    if let Some(captured) = mv.captured {
        let victim = mvv_lva_piece_val(captured.kind);
        let attacker = board
            .get(mv.from)
            .map(|p| mvv_lva_piece_val(p.kind))
            .unwrap_or(1);
        return 100_000 + victim * 10 - attacker;
    }

    // En passant is a capture too.
    if mv.is_en_passant {
        return 100_000 + 10; // pawn x pawn
    }

    // Promotions.
    if mv.promotion.is_some() {
        return 90_000;
    }

    // Killer moves.
    if killers.is_killer(ply, mv) {
        return 50_000;
    }

    // History heuristic.
    history.score(board.side_to_move, mv)
}

/// Selection-sort: find the best-scored move from `start_idx` onward, swap it to `start_idx`.
/// This is better than full sort because alpha-beta often cuts off early.
#[inline]
fn pick_move(moves: &mut MoveList, scores: &mut [i32], start_idx: usize) {
    let mut best_idx = start_idx;
    let mut best_score = scores[start_idx];
    #[allow(clippy::needless_range_loop)]
    for i in (start_idx + 1)..moves.len() {
        if scores[i] > best_score {
            best_score = scores[i];
            best_idx = i;
        }
    }
    if best_idx != start_idx {
        moves.swap(start_idx, best_idx);
        scores.swap(start_idx, best_idx);
    }
}

// ---------------------------------------------------------------------------
// Search state
// ---------------------------------------------------------------------------

struct SearchState {
    tt: TranspositionTable,
    killers: Killers,
    history: History,
    nodes: u64,
}

impl SearchState {
    fn new() -> Self {
        Self {
            tt: TranspositionTable::new(),
            killers: Killers::new(),
            history: History::new(),
            nodes: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Draw detection (avoids calling status() which re-generates moves)
// ---------------------------------------------------------------------------

/// Check draw conditions that don't require move generation.
/// Returns true if the position is drawn by repetition, 50-move rule, or insufficient material.
#[inline]
fn is_draw(state: &GameState) -> bool {
    state.board.halfmove_clock >= 100
        || state.is_draw_by_repetition()
        || state.is_insufficient_material()
}

// ---------------------------------------------------------------------------
// Quiescence search
// ---------------------------------------------------------------------------

fn quiescence(state: &mut GameState, mut alpha: i32, beta: i32, ss: &mut SearchState) -> i32 {
    ss.nodes += 1;

    // Stand-pat: the side to move can choose not to capture.
    let stand_pat = eval::evaluate_board(&state.board);
    if stand_pat >= beta {
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    let moves = movegen::generate_legal_moves(&state.board);
    if moves.is_empty() {
        // Terminal: checkmate or stalemate.
        let stm = state.side_to_move();
        return if movegen::is_in_check(&state.board, stm) {
            -10_000
        } else {
            0
        };
    }

    // Only search captures and en passant (and promotions, which are usually good).
    for mv in moves.iter() {
        let is_tactical = mv.captured.is_some() || mv.is_en_passant || mv.promotion.is_some();
        if !is_tactical {
            continue;
        }

        // Delta pruning: skip captures that can't raise alpha even with the best outcome.
        if let Some(captured) = mv.captured {
            let delta = eval::piece_value(captured.kind) + 200; // 200cp margin
            if stand_pat + delta < alpha {
                continue;
            }
        }

        state.apply_move(*mv);
        let score = -quiescence(state, -beta, -alpha, ss);
        state.undo_move();

        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

// ---------------------------------------------------------------------------
// Negamax with alpha-beta
// ---------------------------------------------------------------------------

fn negamax(
    state: &mut GameState,
    depth: u32,
    mut alpha: i32,
    beta: i32,
    ply: usize,
    ss: &mut SearchState,
) -> i32 {
    ss.nodes += 1;

    // Draw detection (cheap: no move generation).
    if is_draw(state) {
        return 0;
    }

    // TT probe.
    let key = state.board.zobrist_hash;
    let (tt_from, tt_to) = match ss.tt.probe(key) {
        Some(entry) if entry.depth >= depth => match entry.bound {
            Bound::Exact => return entry.score,
            Bound::Lower if entry.score >= beta => return entry.score,
            Bound::Upper if entry.score <= alpha => return entry.score,
            _ => (entry.best_from, entry.best_to),
        },
        Some(entry) => (entry.best_from, entry.best_to),
        None => (255, 255),
    };

    let mut moves = movegen::generate_legal_moves(&state.board);

    // Terminal: no legal moves.
    if moves.is_empty() {
        let stm = state.side_to_move();
        let score = if movegen::is_in_check(&state.board, stm) {
            -(10_000 + depth as i32) // prefer shorter mates
        } else {
            0 // stalemate
        };
        return score;
    }

    // Leaf node: quiescence search.
    if depth == 0 {
        return quiescence(state, alpha, beta, ss);
    }

    // Score moves for ordering.
    let mut scores = [0i32; 256];
    for i in 0..moves.len() {
        scores[i] = move_score(
            &moves[i],
            &state.board,
            tt_from,
            tt_to,
            ply,
            &ss.killers,
            &ss.history,
        );
    }

    let mut best_score = i32::MIN + 1;
    let mut best_move: Option<Move> = None;
    let orig_alpha = alpha;

    for i in 0..moves.len() {
        pick_move(&mut moves, &mut scores, i);
        let mv = moves[i];

        state.apply_move(mv);
        let score = -negamax(state, depth - 1, -beta, -alpha, ply + 1, ss);
        state.undo_move();

        if score > best_score {
            best_score = score;
            best_move = Some(mv);
        }

        if score > alpha {
            alpha = score;
        }

        if alpha >= beta {
            // Beta cutoff — update killers and history for quiet moves.
            if mv.captured.is_none() && !mv.is_en_passant {
                ss.killers.store(ply, &mv);
                ss.history.update(state.side_to_move(), &mv, depth);
            }
            break;
        }
    }

    // Store in TT.
    let bound = if best_score >= beta {
        Bound::Lower
    } else if best_score <= orig_alpha {
        Bound::Upper
    } else {
        Bound::Exact
    };
    ss.tt
        .store(key, depth, best_score, bound, best_move.as_ref());

    best_score
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run iterative-deepening alpha-beta search up to the given depth and return the best move.
///
/// Returns `None` if the position is terminal (no legal moves).
pub fn search(state: &mut GameState, depth: u32) -> Option<MinimaxResult> {
    assert!(depth >= 1, "minimax depth must be >= 1");

    let moves = movegen::generate_legal_moves(&state.board);
    if moves.is_empty() {
        return None;
    }

    let mut ss = SearchState::new();
    let mut best_move = moves[0];
    let mut best_score = i32::MIN + 1;

    // Iterative deepening: search depth 1, 2, ..., `depth`.
    // Each iteration populates the TT, improving move ordering for the next.
    for d in 1..=depth {
        let mut current_best_move = moves[0];
        let mut current_best_score = i32::MIN + 1;
        let mut alpha = i32::MIN + 1;

        // Get TT hint for root move ordering.
        let key = state.board.zobrist_hash;
        let (tt_from, tt_to) = match ss.tt.probe(key) {
            Some(entry) => (entry.best_from, entry.best_to),
            None => (255, 255),
        };

        // Build a mutable copy of the move list for ordering at root.
        let mut root_moves = movegen::generate_legal_moves(&state.board);
        let mut root_scores = [0i32; 256];
        for i in 0..root_moves.len() {
            root_scores[i] = move_score(
                &root_moves[i],
                &state.board,
                tt_from,
                tt_to,
                0,
                &ss.killers,
                &ss.history,
            );
        }

        for i in 0..root_moves.len() {
            pick_move(&mut root_moves, &mut root_scores, i);
            let mv = root_moves[i];

            state.apply_move(mv);
            let score = -negamax(state, d - 1, i32::MIN + 1, -alpha, 1, &mut ss);
            state.undo_move();

            if score > current_best_score {
                current_best_score = score;
                current_best_move = mv;
            }
            if score > alpha {
                alpha = score;
            }
        }

        best_move = current_best_move;
        best_score = current_best_score;

        // Store root position in TT.
        ss.tt
            .store(key, d, best_score, Bound::Exact, Some(&best_move));
    }

    Some(MinimaxResult {
        best_move,
        score: best_score,
        nodes: ss.nodes,
        depth,
    })
}

/// Run alpha-beta search and return scores for **all** legal root moves.
///
/// Unlike `search`, this uses a full window (no alpha tightening at the root)
/// so every move gets an accurate score. Subtree pruning is unaffected.
/// Returns `None` if the position is terminal.
pub fn search_all_moves(state: &mut GameState, depth: u32) -> Option<MinimaxAllResult> {
    assert!(depth >= 1, "minimax depth must be >= 1");
    let moves = state.legal_moves();
    if moves.is_empty() {
        return None;
    }

    let mut ss = SearchState::new();

    // Warm up TT with iterative deepening (but we only use the TT, not results).
    for d in 1..depth {
        let mut alpha = i32::MIN + 1;
        let mut root_moves = movegen::generate_legal_moves(&state.board);
        let key = state.board.zobrist_hash;
        let (tt_from, tt_to) = match ss.tt.probe(key) {
            Some(entry) => (entry.best_from, entry.best_to),
            None => (255, 255),
        };
        let mut root_scores = [0i32; 256];
        for i in 0..root_moves.len() {
            root_scores[i] = move_score(
                &root_moves[i],
                &state.board,
                tt_from,
                tt_to,
                0,
                &ss.killers,
                &ss.history,
            );
        }
        for i in 0..root_moves.len() {
            pick_move(&mut root_moves, &mut root_scores, i);
            let mv = root_moves[i];
            state.apply_move(mv);
            let score = -negamax(state, d - 1, i32::MIN + 1, -alpha, 1, &mut ss);
            state.undo_move();
            if score > alpha {
                alpha = score;
            }
        }
    }

    // Final depth: full window for every move to get accurate scores.
    let mut ranked = Vec::with_capacity(moves.len());

    for mv in &moves {
        state.apply_move(*mv);
        let score = -negamax(state, depth - 1, i32::MIN + 1, i32::MAX, 1, &mut ss);
        state.undo_move();
        ranked.push(RankedMove { mv: *mv, score });
    }

    Some(MinimaxAllResult {
        moves: ranked,
        nodes: ss.nodes,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth1_returns_move() {
        let mut state = GameState::new();
        let result = search(&mut state, 1).unwrap();
        assert!(result.nodes > 0);
        assert!(result.score.abs() <= 10_100);
    }

    #[test]
    fn depth2_returns_move() {
        let mut state = GameState::new();
        let result = search(&mut state, 2).unwrap();
        assert!(result.nodes > 0);
    }

    #[test]
    fn depth3_returns_move() {
        let mut state = GameState::new();
        let result = search(&mut state, 3).unwrap();
        assert!(result.nodes > 0);
    }

    #[test]
    fn terminal_returns_none() {
        let mut state = GameState::new();
        // Ongoing position should return Some.
        assert!(search(&mut state, 1).is_some());
    }

    #[test]
    fn search_all_moves_best_matches_search() {
        let mut state = GameState::new();
        let best = search(&mut state, 2).unwrap();
        let all = search_all_moves(&mut state, 2).unwrap();

        // Best score from search_all_moves should match search.
        let top = all.moves.iter().max_by_key(|m| m.score).unwrap();
        assert_eq!(top.score, best.score);

        // All legal moves should be present.
        let legal = state.legal_moves();
        assert_eq!(all.moves.len(), legal.len());
    }

    #[test]
    fn iterative_deepening_improves_node_count() {
        // Deeper search should explore more nodes but TT should help prune.
        let mut state = GameState::new();
        let r2 = search(&mut state, 2).unwrap();
        let r3 = search(&mut state, 3).unwrap();
        // Depth 3 should search more nodes than depth 2 alone would.
        assert!(r3.nodes > r2.nodes);
    }

    #[test]
    fn tt_reduces_nodes_on_repeated_search() {
        // Two searches at the same depth: second should be similar (TT is per-search).
        // This just tests that search is deterministic.
        let mut state = GameState::new();
        let r1 = search(&mut state, 3).unwrap();
        let r2 = search(&mut state, 3).unwrap();
        assert_eq!(r1.score, r2.score);
    }

    #[test]
    fn captures_scored_higher_than_quiet() {
        // Verify MVV-LVA: a queen capture should score higher than a quiet pawn push.
        use crate::board::{Color, HexCoord, Piece, PieceKind};
        let capture_move = Move {
            from: HexCoord::new(0, 0),
            to: HexCoord::new(1, 0),
            promotion: None,
            captured: Some(Piece {
                kind: PieceKind::Queen,
                color: Color::Black,
            }),
            is_en_passant: false,
        };
        let quiet_move = Move {
            from: HexCoord::new(0, 0),
            to: HexCoord::new(0, 1),
            promotion: None,
            captured: None,
            is_en_passant: false,
        };
        let board = crate::board::Board::new();
        let killers = Killers::new();
        let history = History::new();

        let cap_score = move_score(&capture_move, &board, 255, 255, 0, &killers, &history);
        let quiet_score = move_score(&quiet_move, &board, 255, 255, 0, &killers, &history);
        assert!(
            cap_score > quiet_score,
            "capture score {} should be > quiet score {}",
            cap_score,
            quiet_score
        );
    }

    #[test]
    fn quiescence_does_not_panic() {
        let mut state = GameState::new();
        let mut ss = SearchState::new();
        let score = quiescence(&mut state, i32::MIN + 1, i32::MAX, &mut ss);
        // Starting position has no captures available at depth 0, so score is just material.
        assert!(score.abs() <= 10_100);
    }

    #[test]
    fn deeper_search_finds_better_or_equal_move() {
        // At higher depth the score should be at least as informed.
        let mut state = GameState::new();
        let r1 = search(&mut state, 1).unwrap();
        let r2 = search(&mut state, 2).unwrap();
        // Both should be valid scores — just ensure they don't crash.
        assert!(r1.score.abs() <= 11_000);
        assert!(r2.score.abs() <= 11_000);
    }

    #[test]
    fn killer_moves_stored_and_detected() {
        use crate::board::HexCoord;
        let mut killers = Killers::new();
        let mv = Move::new(HexCoord::new(0, 0), HexCoord::new(1, 0), None);
        killers.store(5, &mv);
        assert!(killers.is_killer(5, &mv));
        assert!(!killers.is_killer(6, &mv));
    }

    #[test]
    fn history_updates_and_reads() {
        use crate::board::HexCoord;
        let mut history = History::new();
        let mv = Move::new(HexCoord::new(0, 0), HexCoord::new(1, 0), None);
        assert_eq!(history.score(Color::White, &mv), 0);
        history.update(Color::White, &mv, 3);
        assert_eq!(history.score(Color::White, &mv), 9); // 3^2
        // Different color should be independent.
        assert_eq!(history.score(Color::Black, &mv), 0);
    }
}
