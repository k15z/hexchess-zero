//! Integration tests for MCTS playing strength.
//!
//! The game-playing tests (mcts_beats_random, more_sims_beats_fewer_sims)
//! are marked #[ignore] because they take ~60s. Run with:
//!   cargo test --test mcts_strength -- --ignored

use hexchess_engine::board::{Board, Color, HexCoord, Piece, PieceKind};
use hexchess_engine::game::GameState;
use hexchess_engine::mcts::{HeuristicEvaluator, MctsSearch};
use rand::prelude::IndexedRandom;

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

/// Play a game: MCTS (with given sims) vs random. Returns the winner color or None for draw.
fn play_mcts_vs_random(mcts_color: Color, mcts_sims: u32) -> Option<Color> {
    let mut game = GameState::new();
    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
    let mut rng = rand::rng();

    for _ in 0..300 {
        if game.is_game_over() {
            break;
        }

        if game.side_to_move() == mcts_color {
            let result = search.search(&game, mcts_sims);
            game.apply_move(result.best_move);
        } else {
            let moves = game.legal_moves();
            if moves.is_empty() {
                break;
            }
            let mv = *moves.choose(&mut rng).unwrap();
            game.apply_move(mv);
        }
    }

    match game.status() {
        hexchess_engine::game::GameStatus::Checkmate(winner) => Some(winner),
        _ => None,
    }
}

/// Play MCTS vs MCTS and return the winner (None = draw).
fn play_mcts_vs_mcts(white_sims: u32, black_sims: u32) -> Option<Color> {
    let mut game = GameState::new();
    let mut white_search = MctsSearch::new(Box::new(HeuristicEvaluator));
    let mut black_search = MctsSearch::new(Box::new(HeuristicEvaluator));

    for _ in 0..300 {
        if game.is_game_over() {
            break;
        }

        let result = if game.side_to_move() == Color::White {
            white_search.search(&game, white_sims)
        } else {
            black_search.search(&game, black_sims)
        };
        game.apply_move(result.best_move);
    }

    match game.status() {
        hexchess_engine::game::GameStatus::Checkmate(winner) => Some(winner),
        _ => None,
    }
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
        legal.iter().any(|m| m.to == HexCoord::new(2, 0) && m.captured.map(|p| p.kind) == Some(PieceKind::Queen)),
        "queen capture must be a legal move in this position",
    );

    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
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

    let checkmate_moves: Vec<_> = legal.iter().filter(|&mv| {
        let mut test = game.clone();
        test.apply_move(*mv);
        test.status() == hexchess_engine::game::GameStatus::Checkmate(Color::White)
    }).collect();

    assert!(!checkmate_moves.is_empty(), "position must have at least one mate-in-1");

    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
    let result = search.search(&game, 500);

    assert!(
        checkmate_moves.iter().any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
        "MCTS should find checkmate in 1! Mate moves: {:?}, MCTS chose: {:?}",
        checkmate_moves,
        result.best_move,
    );
}

// ---------------------------------------------------------------------------
// Test: MCTS (200 sims) beats random player
// ---------------------------------------------------------------------------

#[test]
#[ignore] // slow (~50s): run with `cargo test -- --ignored`
fn mcts_beats_random() {
    let num_games = 20;
    let mut mcts_wins = 0;
    let mut random_wins = 0;
    let mut draws = 0;

    for i in 0..num_games {
        let mcts_color = if i % 2 == 0 { Color::White } else { Color::Black };
        match play_mcts_vs_random(mcts_color, 200) {
            Some(winner) if winner == mcts_color => mcts_wins += 1,
            Some(_) => random_wins += 1,
            None => draws += 1,
        }
    }

    let decided = mcts_wins + random_wins;
    assert!(
        decided > 0 && mcts_wins as f64 / decided as f64 >= 0.7,
        "MCTS should beat random >= 70% of decided games, got {mcts_wins} wins, {random_wins} losses, {draws} draws",
    );
}

// ---------------------------------------------------------------------------
// Test: More simulations beats fewer simulations
// ---------------------------------------------------------------------------

#[test]
#[ignore] // slow (~60s): run with `cargo test -- --ignored`
fn more_sims_beats_fewer_sims() {
    let num_games = 10;
    let mut high_wins = 0;
    let mut low_wins = 0;
    let mut draws = 0;

    for i in 0..num_games {
        let (white_sims, black_sims) = if i % 2 == 0 { (500, 50) } else { (50, 500) };
        let high_is_white = white_sims > black_sims;

        match play_mcts_vs_mcts(white_sims, black_sims) {
            Some(Color::White) if high_is_white => high_wins += 1,
            Some(Color::Black) if !high_is_white => high_wins += 1,
            Some(_) => low_wins += 1,
            None => draws += 1,
        }
    }

    let decided = high_wins + low_wins;
    assert!(
        decided == 0 || high_wins as f64 / decided as f64 >= 0.5,
        "500 sims should beat 50 sims >= 50% of decided games, got {high_wins} wins, {low_wins} losses, {draws} draws",
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
    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
    let result = search.search(&game, 200);
    assert!(result.value > 0.0, "White up a queen should have positive value, got {}", result.value);
}

#[test]
fn mcts_value_negative_when_losing() {
    let board = make_board_with_pieces(&[
        (0, -4, PieceKind::King, Color::White),
        (0, 4, PieceKind::King, Color::Black),
        (0, 3, PieceKind::Queen, Color::Black),
    ]);
    let game = GameState::from_board(board);
    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
    let result = search.search(&game, 200);
    assert!(result.value < 0.0, "White down a queen should have negative value, got {}", result.value);
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
    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
    let result = search.search(&game, 200);
    assert!(result.value > 0.0, "White to move with extra queen should be positive");

    // Black to move, black has extra queen — value should also be positive
    // (value is always from the side-to-move's perspective).
    let mut board2 = make_board_with_pieces(&[
        (0, -4, PieceKind::King, Color::White),
        (0, 4, PieceKind::King, Color::Black),
        (1, 4, PieceKind::Queen, Color::Black),
    ]);
    board2.side_to_move = Color::Black;
    let game2 = GameState::from_board(board2);
    let mut search2 = MctsSearch::new(Box::new(HeuristicEvaluator));
    let result2 = search2.search(&game2, 200);
    assert!(result2.value > 0.0, "Black to move with extra queen should be positive");
}
