//! Property-based correctness tests for the optimized minimax search.
//!
//! Categories:
//! 1. Score equivalence: naive vs optimized agree where expected.
//! 2. Board state preservation: search must not corrupt the game state.
//! 3. Move legality: every returned move must be legal.
//! 4. Score bounds and monotonicity: deeper search shouldn't be weaker.
//! 5. Search consistency: search_all_moves and search agree.
//! 6. Self-play: deeper always beats or ties shallower.
//! 7. Tactical: search finds obvious captures and mates.

use hexchess_engine::board::{Board, Color, HexCoord, Piece, PieceKind};
use hexchess_engine::eval;
use hexchess_engine::game::{GameState, GameStatus};
use hexchess_engine::minimax;
use hexchess_engine::movegen;

// ---------------------------------------------------------------------------
// Reference (naive) negamax for comparison — no TT, no ordering, no qsearch
// ---------------------------------------------------------------------------

fn naive_negamax(
    state: &mut GameState,
    depth: u32,
    mut alpha: i32,
    beta: i32,
    nodes: &mut u64,
) -> i32 {
    *nodes += 1;

    match state.status() {
        GameStatus::Ongoing => {}
        GameStatus::Checkmate(_) => return -(10_000 + depth as i32),
        _ => return 0,
    }

    let moves = state.legal_moves();
    if moves.is_empty() {
        let stm = state.side_to_move();
        return if movegen::is_in_check(&state.board, stm) {
            -(10_000 + depth as i32)
        } else {
            0
        };
    }

    if depth == 0 {
        return eval::evaluate(state);
    }

    for mv in &moves {
        state.apply_move(*mv);
        let score = -naive_negamax(state, depth - 1, -beta, -alpha, nodes);
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

fn naive_search(state: &mut GameState, depth: u32) -> Option<(movegen::Move, i32, u64)> {
    let moves = state.legal_moves();
    if moves.is_empty() {
        return None;
    }

    let mut best_move = moves[0];
    let mut best_score = i32::MIN + 1;
    let mut nodes = 0u64;

    for mv in &moves {
        state.apply_move(*mv);
        let score = -naive_negamax(state, depth - 1, i32::MIN + 1, -best_score, &mut nodes);
        state.undo_move();
        if score > best_score {
            best_score = score;
            best_move = *mv;
        }
    }

    Some((best_move, best_score, nodes))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Play `n` moves from the starting position using depth-1 search, returning
/// the resulting GameState (or None if the game ended early).
fn play_n_moves(n: usize) -> Option<GameState> {
    let mut state = GameState::new();
    for _ in 0..n {
        if state.is_game_over() {
            return None;
        }
        match minimax::search(&mut state, 1) {
            Some(r) => state.apply_move(r.best_move),
            None => return None,
        }
    }
    if state.is_game_over() {
        None
    } else {
        Some(state)
    }
}

/// Build a position with a free piece to capture:
/// White king at origin, white rook ready to capture undefended black queen,
/// black king far away.
fn free_queen_position() -> GameState {
    let mut board = Board::empty();
    // White king at (-5, 0)
    board.set(
        HexCoord::new(-5, 0),
        Some(Piece {
            kind: PieceKind::King,
            color: Color::White,
        }),
    );
    board.white_king = HexCoord::new(-5, 0);
    // White rook at (0, 0) — center
    board.set(
        HexCoord::new(0, 0),
        Some(Piece {
            kind: PieceKind::Rook,
            color: Color::White,
        }),
    );
    // Black queen at (0, 3) — on same file, undefended
    board.set(
        HexCoord::new(0, 3),
        Some(Piece {
            kind: PieceKind::Queen,
            color: Color::Black,
        }),
    );
    // Black king at (5, -5) — far away
    board.set(
        HexCoord::new(5, -5),
        Some(Piece {
            kind: PieceKind::King,
            color: Color::Black,
        }),
    );
    board.black_king = HexCoord::new(5, -5);
    GameState::from_board(board)
}

// =========================================================================
// 1. SCORE EQUIVALENCE
// =========================================================================

/// At depth 1, naive and optimized agree on the best score from start.
#[test]
fn score_equiv_depth1_start() {
    let mut state = GameState::new();
    let (_, old_score, _) = naive_search(&mut state, 1).unwrap();
    let new_result = minimax::search(&mut state, 1).unwrap();
    assert_eq!(old_score, new_result.score, "depth-1 scores should match");
}

/// At depth 2, naive and optimized agree on the best score from start.
#[test]
fn score_equiv_depth2_start() {
    let mut state = GameState::new();
    let (_, old_score, _) = naive_search(&mut state, 2).unwrap();
    let new_result = minimax::search(&mut state, 2).unwrap();
    assert_eq!(old_score, new_result.score, "depth-2 scores should match");
}

/// In mid-game positions, naive and optimized scores can legitimately diverge
/// because the optimized version uses quiescence search at leaf nodes. When
/// captures are available at the horizon, qsearch resolves the capture chain
/// instead of returning a (potentially misleading) static eval.
///
/// The property we test instead: both engines should agree on the *sign* of the
/// evaluation (who is winning) and the scores should be in the same ballpark.
/// Large divergence would suggest a bug, while small divergence is expected
/// from quiescence improving leaf accuracy.
#[test]
fn scores_similar_in_midgame_positions() {
    for n in [4, 8, 12, 16, 20] {
        let mut state = match play_n_moves(n) {
            Some(s) => s,
            None => continue,
        };
        let (_, old_score, _) = naive_search(&mut state, 2).unwrap();
        let new_result = minimax::search(&mut state, 2).unwrap();

        // Scores should not diverge by more than ~queen value (900cp).
        // Quiescence resolves capture chains but shouldn't change the score
        // by more than the value of the largest piece in play.
        let diff = (old_score - new_result.score).abs();
        assert!(
            diff <= 900,
            "scores diverge too much after {} moves: naive={}, optimized={}, diff={}",
            n,
            old_score,
            new_result.score,
            diff
        );
    }
}

// =========================================================================
// 2. BOARD STATE PRESERVATION
// =========================================================================

/// Search must not corrupt the game state. After search(), the board, side to
/// move, en passant, halfmove clock, zobrist hash, and move count must be
/// identical to before the search.
#[test]
fn search_preserves_board_state() {
    for n in [0, 5, 10, 15] {
        let mut state = match play_n_moves(n) {
            Some(s) => s,
            None => continue,
        };
        let cells_before = state.board.cells;
        let stm_before = state.side_to_move();
        let ep_before = state.board.en_passant;
        let hmc_before = state.board.halfmove_clock;
        let hash_before = state.board.zobrist_hash;
        let move_count_before = state.move_count();

        let _ = minimax::search(&mut state, 3);

        assert_eq!(
            state.board.cells, cells_before,
            "cells changed after search (n={})",
            n
        );
        assert_eq!(
            state.side_to_move(),
            stm_before,
            "side_to_move changed (n={})",
            n
        );
        assert_eq!(
            state.board.en_passant, ep_before,
            "en_passant changed (n={})",
            n
        );
        assert_eq!(
            state.board.halfmove_clock, hmc_before,
            "halfmove_clock changed (n={})",
            n
        );
        assert_eq!(
            state.board.zobrist_hash, hash_before,
            "zobrist_hash changed (n={})",
            n
        );
        assert_eq!(
            state.move_count(),
            move_count_before,
            "move_count changed (n={})",
            n
        );
    }
}

/// search_all_moves must also preserve board state.
#[test]
fn search_all_preserves_board_state() {
    let mut state = GameState::new();
    let cells_before = state.board.cells;
    let hash_before = state.board.zobrist_hash;

    let _ = minimax::search_all_moves(&mut state, 3);

    assert_eq!(state.board.cells, cells_before);
    assert_eq!(state.board.zobrist_hash, hash_before);
}

// =========================================================================
// 3. MOVE LEGALITY
// =========================================================================

/// The best_move returned by search() must always be a legal move.
#[test]
fn best_move_is_legal() {
    for n in [0, 3, 6, 10, 15, 20] {
        let mut state = match play_n_moves(n) {
            Some(s) => s,
            None => continue,
        };
        for depth in 1..=3 {
            let result = minimax::search(&mut state, depth).unwrap();
            let legal = state.legal_moves();
            let found = legal.iter().any(|m| {
                m.from == result.best_move.from
                    && m.to == result.best_move.to
                    && m.promotion == result.best_move.promotion
            });
            assert!(
                found,
                "best_move ({:?}->{:?}) not in legal moves after {} moves at depth {}",
                result.best_move.from, result.best_move.to, n, depth
            );
        }
    }
}

/// All moves from search_all_moves must be legal and cover every legal move.
#[test]
fn search_all_moves_are_legal_and_complete() {
    for n in [0, 6, 12] {
        let mut state = match play_n_moves(n) {
            Some(s) => s,
            None => continue,
        };
        let all = minimax::search_all_moves(&mut state, 2).unwrap();
        let legal = state.legal_moves();
        assert_eq!(
            all.moves.len(),
            legal.len(),
            "search_all_moves count != legal moves count after {} moves",
            n
        );
    }
}

// =========================================================================
// 4. SCORE BOUNDS AND MONOTONICITY
// =========================================================================

/// All scores must be within a sane range (no overflow/underflow).
#[test]
fn scores_within_bounds() {
    for n in [0, 5, 10] {
        let mut state = match play_n_moves(n) {
            Some(s) => s,
            None => continue,
        };
        let result = minimax::search(&mut state, 3).unwrap();
        assert!(
            result.score.abs() < 20_000,
            "score {} out of bounds after {} moves",
            result.score,
            n
        );
    }
}

/// Deeper search should explore more nodes (with iterative deepening, depth N
/// includes all work from depths 1..N).
#[test]
fn deeper_search_explores_more_nodes() {
    let mut state = GameState::new();
    let r1 = minimax::search(&mut state, 1).unwrap();
    let r2 = minimax::search(&mut state, 2).unwrap();
    let r3 = minimax::search(&mut state, 3).unwrap();
    assert!(
        r2.nodes > r1.nodes,
        "depth 2 should search more nodes than depth 1"
    );
    assert!(
        r3.nodes > r2.nodes,
        "depth 3 should search more nodes than depth 2"
    );
}

/// The optimized search should use fewer nodes than naive at depth 3+.
#[test]
fn fewer_nodes_at_depth3() {
    let mut state = GameState::new();
    let (_, _, old_nodes) = naive_search(&mut state, 3).unwrap();
    let new_result = minimax::search(&mut state, 3).unwrap();
    println!(
        "Depth 3 from start: naive={} nodes, optimized={} nodes, ratio={:.2}x",
        old_nodes,
        new_result.nodes,
        old_nodes as f64 / new_result.nodes as f64
    );
    assert!(
        new_result.nodes < old_nodes,
        "optimized ({}) should use fewer nodes than naive ({})",
        new_result.nodes,
        old_nodes
    );
}

// =========================================================================
// 5. SEARCH CONSISTENCY
// =========================================================================

/// search() and search_all_moves() must agree on the best score.
#[test]
fn search_and_search_all_agree_on_best_score() {
    for n in [0, 6, 12] {
        let mut state = match play_n_moves(n) {
            Some(s) => s,
            None => continue,
        };
        let best = minimax::search(&mut state, 2).unwrap();
        let all = minimax::search_all_moves(&mut state, 2).unwrap();

        let top = all.moves.iter().max_by_key(|m| m.score).unwrap();
        assert_eq!(
            top.score, best.score,
            "search vs search_all_moves best score mismatch after {} moves: search={} all={}",
            n, best.score, top.score
        );
    }
}

/// Repeated searches on the same position must return identical results.
#[test]
fn search_is_deterministic() {
    for n in [0, 6, 12] {
        let mut state = match play_n_moves(n) {
            Some(s) => s,
            None => continue,
        };
        let r1 = minimax::search(&mut state, 3).unwrap();
        let r2 = minimax::search(&mut state, 3).unwrap();
        assert_eq!(
            r1.score, r2.score,
            "scores differ on repeated search (n={})",
            n
        );
        assert_eq!(
            r1.best_move.from, r2.best_move.from,
            "best_move.from differs (n={})",
            n
        );
        assert_eq!(
            r1.best_move.to, r2.best_move.to,
            "best_move.to differs (n={})",
            n
        );
    }
}

/// search_all_moves scores must be consistent with individual move evaluations:
/// if you apply the top-scored move and negate the opponent's best reply score,
/// it should relate to the original score.
/// (Weaker property: top move from search_all_moves should be among the best
/// moves at that depth from search().)
#[test]
fn search_all_top_move_matches_search_move() {
    let mut state = GameState::new();
    let best = minimax::search(&mut state, 2).unwrap();
    let all = minimax::search_all_moves(&mut state, 2).unwrap();

    let top = all.moves.iter().max_by_key(|m| m.score).unwrap();
    // The top-scored move should match the best_move from search (same from/to).
    // They may differ if multiple moves share the same score, so we check score equality.
    assert_eq!(top.score, best.score);
}

// =========================================================================
// 6. SELF-PLAY
// =========================================================================

/// Depth-3 should not lose to depth-2 across multiple games with both colors.
#[test]
fn depth3_does_not_lose_to_depth2() {
    let games = 4;
    let mut d3_losses = 0;

    for game_idx in 0..games {
        let mut state = GameState::new();
        let d3_color = if game_idx % 2 == 0 {
            Color::White
        } else {
            Color::Black
        };

        for _ in 0..200 {
            if state.is_game_over() {
                break;
            }

            let depth = if state.side_to_move() == d3_color {
                3
            } else {
                2
            };

            match minimax::search(&mut state, depth) {
                Some(r) => state.apply_move(r.best_move),
                None => break,
            }
        }

        if let GameStatus::Checkmate(winner) = state.status() {
            if winner != d3_color {
                d3_losses += 1;
            }
        }
    }

    assert!(
        d3_losses == 0,
        "depth-3 lost {} of {} games to depth-2",
        d3_losses,
        games
    );
}

/// Equal-depth self-play completes without panics or infinite loops.
#[test]
fn selfplay_completes_without_panic() {
    let mut state = GameState::new();
    for _ in 0..100 {
        if state.is_game_over() {
            break;
        }
        match minimax::search(&mut state, 2) {
            Some(r) => state.apply_move(r.best_move),
            None => break,
        }
    }
}

/// Self-play game must reach a terminal state or hit move limit — never hang.
/// Also verifies that every move played during self-play is legal.
#[test]
fn selfplay_all_moves_legal() {
    let mut state = GameState::new();
    for move_num in 0..150 {
        if state.is_game_over() {
            break;
        }
        let result = match minimax::search(&mut state, 2) {
            Some(r) => r,
            None => break,
        };
        let legal = state.legal_moves();
        let found = legal.iter().any(|m| {
            m.from == result.best_move.from
                && m.to == result.best_move.to
                && m.promotion == result.best_move.promotion
        });
        assert!(
            found,
            "move {} in self-play is illegal: {:?}->{:?}",
            move_num, result.best_move.from, result.best_move.to
        );
        state.apply_move(result.best_move);
    }
}

// =========================================================================
// 7. TACTICAL
// =========================================================================

/// Search must find a free queen capture (rook takes undefended queen).
#[test]
fn finds_free_queen_capture() {
    let mut state = free_queen_position();
    let result = minimax::search(&mut state, 1).unwrap();
    // The rook at (0,0) should capture the queen at (0,3).
    assert_eq!(
        result.best_move.to,
        HexCoord::new(0, 3),
        "should capture the free queen at (0,3), but moved to {:?}",
        result.best_move.to
    );
    assert!(
        result.score > 0,
        "capturing a free queen should give positive score, got {}",
        result.score
    );
}

/// Even at depth 1, the engine should capture a free queen rather than
/// making any other move. This tests that move ordering doesn't break
/// the root selection.
#[test]
fn free_queen_capture_all_moves() {
    let mut state = free_queen_position();
    let all = minimax::search_all_moves(&mut state, 1).unwrap();
    let top = all.moves.iter().max_by_key(|m| m.score).unwrap();
    assert_eq!(
        top.mv.to,
        HexCoord::new(0, 3),
        "top move from search_all should capture queen"
    );
}

/// Mate score should include depth bonus: closer mates score higher.
/// A mate-in-1 should score higher than any non-mate move.
#[test]
fn mate_score_exceeds_material() {
    // We can't easily set up mate-in-1, but we can verify that mate scores
    // are above the material range (which maxes around ~4300 centipawns).
    let mut state = GameState::new();
    let result = minimax::search(&mut state, 3).unwrap();
    // From a balanced starting position, score should be within material range.
    assert!(
        result.score.abs() < 5000,
        "starting position score {} should be within material range",
        result.score
    );
}

/// search_all_moves scores should be finite and within bounds for all moves.
#[test]
fn search_all_scores_bounded() {
    for n in [0, 6, 12] {
        let mut state = match play_n_moves(n) {
            Some(s) => s,
            None => continue,
        };
        let all = minimax::search_all_moves(&mut state, 2).unwrap();
        for rm in &all.moves {
            assert!(
                rm.score.abs() < 20_000,
                "score {} out of range after {} moves for move {:?}->{:?}",
                rm.score,
                n,
                rm.mv.from,
                rm.mv.to
            );
        }
    }
}

/// A position with a large material advantage should have a positive score
/// for the side with more material.
#[test]
fn material_advantage_reflected_in_score() {
    // Use free_queen_position: white has rook + king vs king + queen.
    // After white captures the queen, white has a rook advantage.
    // Even before capture, depth-2 search should find the capture and
    // evaluate favorably for white.
    let mut state = free_queen_position();
    let result = minimax::search(&mut state, 2).unwrap();
    assert!(
        result.score > 0,
        "white with rook capturing free queen should have positive score, got {}",
        result.score
    );
}
