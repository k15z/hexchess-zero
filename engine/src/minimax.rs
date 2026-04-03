//! Alpha-beta minimax search for hexagonal chess.
//!
//! Optimizations:
//! - Iterative deepening with transposition table
//! - MVV-LVA capture ordering
//! - Killer move heuristic (2 per ply)
//! - History heuristic for quiet moves
//! - Quiescence search (captures only at horizon, depth-capped)
//! - No redundant move generation (avoids double `status()` calls)

use crate::board::{Color, NUM_CELLS, PieceKind};
use crate::eval;
use crate::game::GameState;
use crate::movegen::{self, Move, MoveList};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MATE_SCORE: i32 = 10_000;
const MAX_PLY: usize = 128;
const TT_SIZE: usize = 1 << 20; // ~1M entries
const TT_MASK: usize = TT_SIZE - 1;
const MAX_QS_DEPTH: u32 = 8;

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

/// Result of `search_with_policy`: best move from full search + scores for
/// all root moves from a shallow TT-backed re-search.
pub struct MinimaxPolicyResult {
    pub best_move: Move,
    pub best_score: i32,
    pub move_scores: Vec<RankedMove>,
    pub nodes: u64,
}

// ---------------------------------------------------------------------------
// Mate score adjustment for TT storage
// ---------------------------------------------------------------------------

/// Convert a score to TT-relative form. Mate scores are stored relative to
/// the current node so they remain correct when probed at a different ply.
#[inline]
fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_SCORE - MAX_PLY as i32 {
        score + ply as i32
    } else if score <= -(MATE_SCORE - MAX_PLY as i32) {
        score - ply as i32
    } else {
        score
    }
}

/// Convert a TT-stored score back to a search-relative score at `ply`.
#[inline]
fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_SCORE - MAX_PLY as i32 {
        score - ply as i32
    } else if score <= -(MATE_SCORE - MAX_PLY as i32) {
        score + ply as i32
    } else {
        score
    }
}

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower, // beta cutoff
    Upper, // failed low
}

#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    depth: u32,
    score: i32, // stored in TT-relative form (mate scores adjusted)
    bound: Bound,
    best_from: u8,  // cell index, 255 = none
    best_to: u8,    // cell index
    best_promo: u8, // 0=none, 1=queen, 2=rook, 3=bishop, 4=knight
}

impl TtEntry {
    const EMPTY: Self = Self {
        key: 0,
        depth: 0,
        score: 0,
        bound: Bound::Exact,
        best_from: 255,
        best_to: 255,
        best_promo: 0,
    };
}

#[inline]
fn promo_to_u8(p: Option<PieceKind>) -> u8 {
    match p {
        None => 0,
        Some(PieceKind::Queen) => 1,
        Some(PieceKind::Rook) => 2,
        Some(PieceKind::Bishop) => 3,
        Some(PieceKind::Knight) => 4,
        _ => 0,
    }
}

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
        if existing.key != key || depth >= existing.depth {
            let (bf, bt, bp) = match best_move {
                Some(mv) => (
                    crate::board::coord_to_index(mv.from).unwrap_or(255) as u8,
                    crate::board::coord_to_index(mv.to).unwrap_or(255) as u8,
                    promo_to_u8(mv.promotion),
                ),
                None => (255, 255, 0),
            };
            self.entries[idx] = TtEntry {
                key,
                depth,
                score,
                bound,
                best_from: bf,
                best_to: bt,
                best_promo: bp,
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Killer moves & history heuristic
// ---------------------------------------------------------------------------

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
        let pair = move_to_pair(mv);
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
        let pair = move_to_pair(mv);
        self.slots[ply][0] == pair || self.slots[ply][1] == pair
    }
}

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
        let (fi, ti) = move_indices(mv);
        self.table[color as usize][fi][ti] += (depth * depth) as i32;
    }

    #[inline]
    fn score(&self, color: Color, mv: &Move) -> i32 {
        let (fi, ti) = move_indices(mv);
        self.table[color as usize][fi][ti]
    }
}

/// Convert move from/to to cell index pair for killer/history tables.
#[inline]
fn move_to_pair(mv: &Move) -> (u8, u8) {
    (
        crate::board::coord_to_index(mv.from).unwrap_or(255) as u8,
        crate::board::coord_to_index(mv.to).unwrap_or(255) as u8,
    )
}

/// Convert move from/to to cell index pair (usize) for history table.
#[inline]
fn move_indices(mv: &Move) -> (usize, usize) {
    (
        crate::board::coord_to_index(mv.from).unwrap_or(0),
        crate::board::coord_to_index(mv.to).unwrap_or(0),
    )
}

// ---------------------------------------------------------------------------
// Move ordering
// ---------------------------------------------------------------------------

#[inline]
fn mvv_lva_piece_val(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::Pawn => 1,
        PieceKind::Knight => 3,
        PieceKind::Bishop => 3,
        PieceKind::Rook => 5,
        PieceKind::Queen => 9,
        PieceKind::King => 100,
    }
}

/// TT move hint for move ordering: (from_cell_idx, to_cell_idx, promotion).
type TtHint = (u8, u8, u8);

const NO_TT_HINT: TtHint = (255, 255, 0);

/// Score a move for ordering. Higher is searched first.
#[inline]
fn move_score(
    mv: &Move,
    board: &crate::board::Board,
    tt_hint: TtHint,
    ply: usize,
    killers: &Killers,
    history: &History,
) -> i32 {
    let (fi, ti) = move_to_pair(mv);
    let (tt_from, tt_to, tt_promo) = tt_hint;

    if fi == tt_from && ti == tt_to && tt_from != 255 && promo_to_u8(mv.promotion) == tt_promo {
        return 1_000_000;
    }

    if let Some(captured) = mv.captured {
        let victim = mvv_lva_piece_val(captured.kind);
        let attacker = board
            .get(mv.from)
            .map(|p| mvv_lva_piece_val(p.kind))
            .unwrap_or(1);
        return 100_000 + victim * 10 - attacker;
    }

    if mv.is_en_passant {
        return 100_000 + 10;
    }

    if mv.promotion.is_some() {
        return 90_000;
    }

    if killers.is_killer(ply, mv) {
        return 50_000;
    }

    history.score(board.side_to_move, mv)
}

/// Selection-sort: pick the best-scored move from `start_idx` onward.
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

/// Controls leaf evaluation and TT write behavior for negamax.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SearchMode {
    /// Full search: quiescence at leaves, killer/history updates, TT writes.
    Full,
    /// TT-probe-only: static eval at leaves, no TT writes. Used by Phase 2
    /// of `search_with_policy()` to avoid polluting Phase 1's TT entries.
    ProbeOnly,
}

// ---------------------------------------------------------------------------
// Quiescence search (depth-capped)
// ---------------------------------------------------------------------------

fn quiescence(
    state: &mut GameState,
    mut alpha: i32,
    beta: i32,
    qs_depth: u32,
    ss: &mut SearchState,
) -> i32 {
    ss.nodes += 1;

    let stand_pat = eval::evaluate_board(&state.board);
    if stand_pat >= beta {
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    if qs_depth == 0 {
        return alpha;
    }

    let in_check = movegen::is_in_check(&state.board, state.side_to_move());

    if in_check {
        let moves = movegen::generate_legal_moves(&state.board);
        if moves.is_empty() {
            return -(MATE_SCORE);
        }
        for mv in moves.iter() {
            state.apply_move(*mv);
            let score = -quiescence(state, -beta, -alpha, qs_depth - 1, ss);
            state.undo_move();

            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
        return alpha;
    }

    let moves = movegen::generate_legal_moves(&state.board);

    for mv in moves.iter() {
        let is_tactical = mv.captured.is_some() || mv.is_en_passant || mv.promotion.is_some();
        if !is_tactical {
            continue;
        }

        if let Some(captured) = mv.captured {
            let delta = eval::piece_value(captured.kind) + 200;
            if stand_pat + delta < alpha {
                continue;
            }
        }

        state.apply_move(*mv);
        let score = -quiescence(state, -beta, -alpha, qs_depth - 1, ss);
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
    mode: SearchMode,
    ss: &mut SearchState,
) -> i32 {
    ss.nodes += 1;

    // Draw detection must come before TT cutoffs: repetition is path-dependent,
    // so a TT entry from a non-repeating path would return a wrong score if
    // the current path is a threefold repetition.
    if state.is_draw() {
        return 0;
    }

    let key = state.board.zobrist_hash;
    let tt_hint = match ss.tt.probe(key) {
        Some(entry) if entry.depth >= depth => {
            let score = score_from_tt(entry.score, ply);
            match entry.bound {
                Bound::Exact => return score,
                Bound::Lower if score >= beta => return score,
                Bound::Upper if score <= alpha => return score,
                _ => (entry.best_from, entry.best_to, entry.best_promo),
            }
        }
        Some(entry) => (entry.best_from, entry.best_to, entry.best_promo),
        None => NO_TT_HINT,
    };

    let mut moves = movegen::generate_legal_moves(&state.board);

    if moves.is_empty() {
        let stm = state.side_to_move();
        return if movegen::is_in_check(&state.board, stm) {
            -(MATE_SCORE + depth as i32)
        } else {
            0
        };
    }

    if depth == 0 {
        return match mode {
            SearchMode::Full => quiescence(state, alpha, beta, MAX_QS_DEPTH, ss),
            SearchMode::ProbeOnly => eval::evaluate_board(&state.board),
        };
    }

    let mut scores = [0i32; 256];
    for i in 0..moves.len() {
        scores[i] = move_score(
            &moves[i],
            &state.board,
            tt_hint,
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
        let score = -negamax(state, depth - 1, -beta, -alpha, ply + 1, mode, ss);
        state.undo_move();

        if score > best_score {
            best_score = score;
            best_move = Some(mv);
        }

        if score > alpha {
            alpha = score;
        }

        if alpha >= beta {
            if mode == SearchMode::Full && mv.captured.is_none() && !mv.is_en_passant {
                ss.killers.store(ply, &mv);
                ss.history.update(state.side_to_move(), &mv, depth);
            }
            break;
        }
    }

    if mode == SearchMode::Full {
        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_score <= orig_alpha {
            Bound::Upper
        } else {
            Bound::Exact
        };
        ss.tt.store(
            key,
            depth,
            score_to_tt(best_score, ply),
            bound,
            best_move.as_ref(),
        );
    }

    best_score
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Run one iteration of root search at the given depth.
/// Returns `None` if there are no legal moves (terminal position).
fn search_root(state: &mut GameState, d: u32, ss: &mut SearchState) -> Option<(Move, i32)> {
    let key = state.board.zobrist_hash;
    let tt_hint = match ss.tt.probe(key) {
        Some(entry) => (entry.best_from, entry.best_to, entry.best_promo),
        None => NO_TT_HINT,
    };

    let mut root_moves = movegen::generate_legal_moves(&state.board);
    if root_moves.is_empty() {
        return None;
    }

    let mut root_scores = [0i32; 256];
    for i in 0..root_moves.len() {
        root_scores[i] = move_score(
            &root_moves[i],
            &state.board,
            tt_hint,
            0,
            &ss.killers,
            &ss.history,
        );
    }

    let mut best_move = root_moves[0];
    let mut best_score = i32::MIN + 1;
    let mut alpha = i32::MIN + 1;

    for i in 0..root_moves.len() {
        pick_move(&mut root_moves, &mut root_scores, i);
        let mv = root_moves[i];

        state.apply_move(mv);
        let score = -negamax(state, d - 1, i32::MIN + 1, -alpha, 1, SearchMode::Full, ss);
        state.undo_move();

        if score > best_score {
            best_score = score;
            best_move = mv;
        }
        if score > alpha {
            alpha = score;
        }
    }

    // All root moves are visited (no beta cutoff), so the best score is exact.
    ss.tt
        .store(key, d, best_score, Bound::Exact, Some(&best_move));

    Some((best_move, best_score))
}

/// Run iterative deepening from depth 1 to `depth`, returning the best move
/// and score. Returns `None` if the position is terminal.
fn iterative_deepen(
    state: &mut GameState,
    depth: u32,
    ss: &mut SearchState,
) -> Option<(Move, i32)> {
    let mut best_move = None;
    let mut best_score = i32::MIN + 1;

    for d in 1..=depth {
        match search_root(state, d, ss) {
            Some((mv, score)) => {
                best_move = Some(mv);
                best_score = score;
            }
            None => return None,
        }
    }

    best_move.map(|mv| (mv, best_score))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run iterative-deepening alpha-beta search up to the given depth.
///
/// Returns `None` if the position is terminal (no legal moves).
pub fn search(state: &mut GameState, depth: u32) -> Option<MinimaxResult> {
    assert!(depth >= 1, "minimax depth must be >= 1");

    let mut ss = SearchState::new();
    let (mv, score) = iterative_deepen(state, depth, &mut ss)?;

    Some(MinimaxResult {
        best_move: mv,
        score,
        nodes: ss.nodes,
        depth,
    })
}

/// Run alpha-beta search and return scores for **all** legal root moves.
///
/// Uses iterative deepening to populate the TT, then scores each root move
/// with a full window at the final depth (including quiescence). Kept for
/// backward compatibility and as a test reference.
/// Returns `None` if the position is terminal.
pub fn search_all_moves(state: &mut GameState, depth: u32) -> Option<MinimaxAllResult> {
    assert!(depth >= 1, "minimax depth must be >= 1");

    let moves = movegen::generate_legal_moves(&state.board);
    if moves.is_empty() {
        return None;
    }

    let mut ss = SearchState::new();

    for d in 1..depth {
        search_root(state, d, &mut ss);
    }

    let mut ranked = Vec::with_capacity(moves.len());
    for mv in moves.iter() {
        state.apply_move(*mv);
        let score = -negamax(
            state,
            depth - 1,
            i32::MIN + 1,
            i32::MAX,
            1,
            SearchMode::Full,
            &mut ss,
        );
        state.undo_move();
        ranked.push(RankedMove { mv: *mv, score });
    }

    Some(MinimaxAllResult {
        moves: ranked,
        nodes: ss.nodes,
    })
}

/// Two-phase search that returns both the best move (from a fully-optimized
/// search) and scores for all root moves (from a shallow TT-backed re-search).
///
/// Phase 1: Run iterative-deepening alpha-beta search with all optimizations
/// (TT, move ordering, quiescence, killers, history) to find the best move
/// and populate the transposition table.
///
/// Phase 2: For each legal root move, run negamax at `depth - 1` with **no
/// quiescence** and in TT-probe-only mode (reads Phase 1's TT entries but
/// does not write, avoiding pollution of quiescence-backed scores with weaker
/// static-eval scores). The TT from Phase 1 short-circuits most subtrees,
/// keeping Phase 2 cheap while avoiding the node blowup that quiescence
/// causes under full-window search.
///
/// The best move's score in `move_scores` is overridden with the exact Phase 1
/// score, since Phase 2 is shallower.
///
/// Returns `None` if the position is terminal.
pub fn search_with_policy(state: &mut GameState, depth: u32) -> Option<MinimaxPolicyResult> {
    assert!(depth >= 1, "minimax depth must be >= 1");

    let mut ss = SearchState::new();

    // Phase 1: full optimized search.
    let (best_move, best_score) = iterative_deepen(state, depth, &mut ss)?;

    // Phase 2: shallow re-search each root move without quiescence.
    // ProbeOnly mode reads Phase 1's TT but doesn't write, preserving the
    // quiescence-backed entries from Phase 1.
    let policy_depth = if depth >= 2 { depth - 1 } else { 1 };
    let moves = movegen::generate_legal_moves(&state.board);
    let mut move_scores = Vec::with_capacity(moves.len());

    for mv in moves.iter() {
        state.apply_move(*mv);
        let score = -negamax(
            state,
            policy_depth - 1,
            i32::MIN + 1,
            i32::MAX,
            1,
            SearchMode::ProbeOnly,
            &mut ss,
        );
        state.undo_move();
        move_scores.push(RankedMove { mv: *mv, score });
    }

    // Override the best move's Phase 2 score with the exact Phase 1 score.
    for rm in &mut move_scores {
        if rm.mv.from == best_move.from
            && rm.mv.to == best_move.to
            && rm.mv.promotion == best_move.promotion
        {
            rm.score = best_score;
            break;
        }
    }

    Some(MinimaxPolicyResult {
        best_move,
        best_score,
        move_scores,
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
        assert!(search(&mut state, 1).is_some());
    }

    #[test]
    fn search_all_moves_best_matches_search() {
        let mut state = GameState::new();
        let best = search(&mut state, 2).unwrap();
        let all = search_all_moves(&mut state, 2).unwrap();

        let top = all.moves.iter().max_by_key(|m| m.score).unwrap();
        assert_eq!(top.score, best.score);

        let legal = state.legal_moves();
        assert_eq!(all.moves.len(), legal.len());
    }

    #[test]
    fn iterative_deepening_improves_node_count() {
        let mut state = GameState::new();
        let r2 = search(&mut state, 2).unwrap();
        let r3 = search(&mut state, 3).unwrap();
        assert!(r3.nodes > r2.nodes);
    }

    #[test]
    fn tt_reduces_nodes_on_repeated_search() {
        let mut state = GameState::new();
        let r1 = search(&mut state, 3).unwrap();
        let r2 = search(&mut state, 3).unwrap();
        assert_eq!(r1.score, r2.score);
    }

    #[test]
    fn captures_scored_higher_than_quiet() {
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

        let cap_score = move_score(&capture_move, &board, NO_TT_HINT, 0, &killers, &history);
        let quiet_score = move_score(&quiet_move, &board, NO_TT_HINT, 0, &killers, &history);
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
        let score = quiescence(&mut state, i32::MIN + 1, i32::MAX, MAX_QS_DEPTH, &mut ss);
        assert!(score.abs() <= 10_100);
    }

    #[test]
    fn quiescence_depth_cap_terminates() {
        let mut state = GameState::new();
        let mut ss = SearchState::new();
        let score = quiescence(&mut state, i32::MIN + 1, i32::MAX, 1, &mut ss);
        assert!(score.abs() <= 10_100);
    }

    #[test]
    fn deeper_search_finds_better_or_equal_move() {
        let mut state = GameState::new();
        let r1 = search(&mut state, 1).unwrap();
        let r2 = search(&mut state, 2).unwrap();
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
        assert_eq!(history.score(Color::Black, &mv), 0);
    }

    #[test]
    fn mate_score_tt_roundtrip() {
        let mate_in_3 = -(MATE_SCORE + 3);
        let stored = score_to_tt(mate_in_3, 5);
        let retrieved = score_from_tt(stored, 5);
        assert_eq!(retrieved, mate_in_3);

        let retrieved_at_2 = score_from_tt(stored, 2);
        assert_eq!(retrieved_at_2, mate_in_3 - 3);
    }

    #[test]
    fn non_mate_score_tt_roundtrip() {
        let score = 150;
        let stored = score_to_tt(score, 7);
        let retrieved = score_from_tt(stored, 3);
        assert_eq!(retrieved, score);
    }

    #[test]
    fn search_with_policy_returns_all_moves() {
        let mut state = GameState::new();
        let result = search_with_policy(&mut state, 2).unwrap();
        let legal = state.legal_moves();
        assert_eq!(result.move_scores.len(), legal.len());
    }

    #[test]
    fn search_with_policy_best_matches_search() {
        let mut state = GameState::new();
        let best = search(&mut state, 3).unwrap();
        let policy = search_with_policy(&mut state, 3).unwrap();
        assert_eq!(
            policy.best_score, best.score,
            "search_with_policy best_score {} != search score {}",
            policy.best_score, best.score
        );
        assert_eq!(policy.best_move.from, best.best_move.from);
        assert_eq!(policy.best_move.to, best.best_move.to);
    }

    #[test]
    fn search_with_policy_preserves_board_state() {
        let mut state = GameState::new();
        let cells_before = state.board.cells;
        let hash_before = state.board.zobrist_hash;
        let stm_before = state.side_to_move();

        let _ = search_with_policy(&mut state, 3);

        assert_eq!(state.board.cells, cells_before);
        assert_eq!(state.board.zobrist_hash, hash_before);
        assert_eq!(state.side_to_move(), stm_before);
    }
}
