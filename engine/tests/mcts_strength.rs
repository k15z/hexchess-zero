//! MCTS tactical tests: unit-level tests that verify MCTS finds obvious
//! tactical moves (captures, mates) and that value estimates have correct
//! sign/perspective.
//!
//! Strength tests (mcts-vs-random, more-sims-vs-fewer) have been removed
//! in favor of the paired-opening arena in Tier 2.

use hexchess_engine::board::{Board, Color, HexCoord, Piece, PieceKind};
use hexchess_engine::game::GameState;
use hexchess_engine::mcts::{HeuristicEvaluator, MctsSearch};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_board_with_pieces(pieces: &[(i8, i8, PieceKind, Color)]) -> Board {
    let mut board = Board::empty();
    for &(q, r, kind, color) in pieces {
        let coord = HexCoord::new(q, r);
        board.set(coord, Some(Piece::new(kind, color)));
        if kind == PieceKind::King {
            match color {
                Color::White => board.white_king = coord,
                Color::Black => board.black_king = coord,
            }
        }
    }
    board
}

// ---------------------------------------------------------------------------
// Test: MCTS finds free queen capture
// ---------------------------------------------------------------------------

#[test]
fn mcts_captures_free_queen() {
    // White rook at (2,-3) can capture undefended black queen at (2,0) along the r-axis.
    let board = make_board_with_pieces(&[
        (0, -4, PieceKind::King, Color::White),
        (2, -3, PieceKind::Rook, Color::White),
        (0, 4, PieceKind::King, Color::Black),
        (2, 0, PieceKind::Queen, Color::Black),
    ]);

    let game = GameState::from_board(board);
    let legal = game.legal_moves();
    assert!(
        legal
            .iter()
            .any(|m| m.to == HexCoord::new(2, 0)
                && m.captured.map(|p| p.kind) == Some(PieceKind::Queen)),
        "queen capture must be a legal move in this position",
    );

    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator::default()));
    let result = search.search(&game, 500);

    assert_eq!(
        result.best_move.to,
        HexCoord::new(2, 0),
        "MCTS should capture the free queen! Instead chose {:?}",
        result.best_move
    );
}

// ---------------------------------------------------------------------------
// Test: MCTS finds checkmate in 1
// ---------------------------------------------------------------------------

#[test]
fn mcts_finds_checkmate_in_1() {
    // Black king cornered at (5,-5). White queen at (4,-2) has 5 checkmate moves.
    let board = make_board_with_pieces(&[
        (3, -5, PieceKind::King, Color::White),
        (4, -2, PieceKind::Queen, Color::White),
        (5, -5, PieceKind::King, Color::Black),
    ]);

    let game = GameState::from_board(board);
    let legal = game.legal_moves();

    let checkmate_moves: Vec<_> = legal
        .iter()
        .filter(|&mv| {
            let mut test = game.clone();
            test.apply_move(*mv);
            test.status() == hexchess_engine::game::GameStatus::Checkmate(Color::White)
        })
        .collect();

    assert!(
        !checkmate_moves.is_empty(),
        "position must have at least one mate-in-1"
    );

    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator::default()));
    // Strict tactical test → use eval config (no Dirichlet noise, no shuffling).
    // The training default injects noise into the root prior which makes
    // mate-finding probabilistic at low sim counts; eval config is what
    // production uses for match play and benchmark suites.
    *search.config_mut() = hexchess_engine::mcts::SearchConfig::eval();
    search.set_rng_seed(0xC0DE_BEEF);
    let result = search.search(&game, 500);

    assert!(
        checkmate_moves
            .iter()
            .any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
        "MCTS should find checkmate in 1! Mate moves: {:?}, MCTS chose: {:?}",
        checkmate_moves,
        result.best_move,
    );
}

// ---------------------------------------------------------------------------
// Test: MCTS value estimate reflects material
// ---------------------------------------------------------------------------

#[test]
fn mcts_value_reflects_material_advantage() {
    let board = make_board_with_pieces(&[
        (0, -4, PieceKind::King, Color::White),
        (0, -3, PieceKind::Queen, Color::White),
        (0, 4, PieceKind::King, Color::Black),
    ]);
    let game = GameState::from_board(board);
    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator::default()));
    let result = search.search(&game, 200);
    assert!(
        result.value > 0.0,
        "White up a queen should have positive value, got {}",
        result.value
    );
}

#[test]
fn mcts_value_negative_when_losing() {
    let board = make_board_with_pieces(&[
        (0, -4, PieceKind::King, Color::White),
        (0, 4, PieceKind::King, Color::Black),
        (0, 3, PieceKind::Queen, Color::Black),
    ]);
    let game = GameState::from_board(board);
    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator::default()));
    let result = search.search(&game, 200);
    assert!(
        result.value < 0.0,
        "White down a queen should have negative value, got {}",
        result.value
    );
}

// ---------------------------------------------------------------------------
// Test: root value perspective is correct for both sides
// ---------------------------------------------------------------------------

#[test]
fn root_qvalue_perspective_is_side_to_move() {
    // White to move, white has extra queen — value should be positive.
    let mut board = make_board_with_pieces(&[
        (0, -4, PieceKind::King, Color::White),
        (1, -4, PieceKind::Queen, Color::White),
        (0, 4, PieceKind::King, Color::Black),
    ]);
    board.side_to_move = Color::White;
    let game = GameState::from_board(board);
    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator::default()));
    let result = search.search(&game, 200);
    assert!(
        result.value > 0.0,
        "White to move with extra queen should be positive"
    );

    // Black to move, black has extra queen — value should also be positive
    // (value is always from the side-to-move's perspective).
    let mut board2 = make_board_with_pieces(&[
        (0, -4, PieceKind::King, Color::White),
        (0, 4, PieceKind::King, Color::Black),
        (1, 4, PieceKind::Queen, Color::Black),
    ]);
    board2.side_to_move = Color::Black;
    let game2 = GameState::from_board(board2);
    let mut search2 = MctsSearch::new(Box::new(HeuristicEvaluator::default()));
    let result2 = search2.search(&game2, 200);
    assert!(
        result2.value > 0.0,
        "Black to move with extra queen should be positive"
    );
}
