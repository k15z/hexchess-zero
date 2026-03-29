//! Validation suite for hexchess engine correctness and MCTS quality.
//!
//! Run with: cargo run --bin validate -p hexchess-engine
//!
//! Tests:
//!   1. Perft — counts leaf nodes at each depth to validate move generation
//!   2. Apply/undo consistency — verifies zobrist hash round-trips
//!   3. Heuristic vs Random — plays games to confirm the heuristic evaluator beats random

use hexchess_engine::board::{Board, Color, HexCoord, Piece, PieceKind};
use hexchess_engine::game::{GameState, GameStatus};
use hexchess_engine::mcts::{Evaluator, HeuristicEvaluator, MctsSearch};
use hexchess_engine::movegen;
use hexchess_engine::serialization;

use rand::Rng;

// ===========================================================================
// 1. PERFT — move generation correctness
// ===========================================================================

fn perft(state: &mut GameState, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = state.legal_moves();
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut count = 0u64;
    for mv in moves {
        state.apply_move(mv);
        count += perft(state, depth - 1);
        state.undo_move();
    }
    count
}

fn run_perft_tests() {
    println!("=== PERFT TESTS (move generation correctness) ===\n");

    let mut state = GameState::new();

    // Depth 1: should match legal move count from starting position
    let d1 = perft(&mut state, 1);
    println!("  Depth 1: {} nodes", d1);
    assert_eq!(d1, 51, "Starting position should have 51 legal moves");

    // Depth 2: opponent's responses
    let d2 = perft(&mut state, 2);
    println!("  Depth 2: {} nodes", d2);
    // Black also has 51 moves from starting position (symmetric)
    // But each of White's 51 moves changes the position, so total won't be 51*51
    // We just verify it's reasonable and consistent
    assert!(d2 > 2000, "Depth 2 should have >2000 nodes, got {}", d2);

    // Depth 3: a deeper check
    let d3 = perft(&mut state, 3);
    println!("  Depth 3: {} nodes", d3);
    assert!(d3 > 100_000, "Depth 3 should have >100k nodes, got {}", d3);

    // Depth 4: even deeper (may take a few seconds)
    let d4 = perft(&mut state, 4);
    println!("  Depth 4: {} nodes", d4);
    assert!(d4 > 5_000_000, "Depth 4 should have >5M nodes, got {}", d4);

    println!("\n  Perft consistency check: running depth 3 twice...");
    let d3_again = perft(&mut state, 3);
    assert_eq!(d3, d3_again, "Perft should be deterministic");
    println!("  OK — deterministic\n");

    println!("  PERFT PASSED\n");
}

// ===========================================================================
// 2. APPLY/UNDO CONSISTENCY — zobrist hash round-trips
// ===========================================================================

fn run_apply_undo_tests() {
    println!("=== APPLY/UNDO CONSISTENCY TESTS ===\n");

    let mut state = GameState::new();
    let initial_hash = state.board.zobrist_hash;
    let initial_cells = state.board.cells;

    // Play 200 random moves, then undo them all
    let mut rng = rand::rng();
    let mut moves_played = Vec::new();

    for i in 0..200 {
        let moves = state.legal_moves();
        if moves.is_empty() {
            println!("  Game ended after {} moves ({})", i, state.status().as_str());
            break;
        }
        let idx = rng.random_range(0..moves.len());
        let mv = moves[idx];
        state.apply_move(mv);
        moves_played.push(mv);
    }

    println!("  Played {} moves, now undoing all...", moves_played.len());

    for _ in 0..moves_played.len() {
        state.undo_move();
    }

    assert_eq!(state.board.zobrist_hash, initial_hash,
        "Zobrist hash should be restored after undo");
    assert_eq!(state.board.cells, initial_cells,
        "Board cells should be restored after undo");
    assert_eq!(state.board.side_to_move, Color::White,
        "Side to move should be White after full undo");

    println!("  Hash, cells, and side_to_move all restored correctly");

    // Run this test multiple times with different random seeds
    for trial in 0..5 {
        let mut state = GameState::new();
        let hash_before = state.board.zobrist_hash;
        let cells_before = state.board.cells;

        let mut count = 0;
        for _ in 0..100 {
            let moves = state.legal_moves();
            if moves.is_empty() { break; }
            let idx = rng.random_range(0..moves.len());
            state.apply_move(moves[idx]);
            count += 1;
        }
        for _ in 0..count {
            state.undo_move();
        }
        assert_eq!(state.board.zobrist_hash, hash_before,
            "Trial {}: hash mismatch after undo", trial);
        assert_eq!(state.board.cells, cells_before,
            "Trial {}: cells mismatch after undo", trial);
    }
    println!("  5 additional random trials passed");

    println!("\n  APPLY/UNDO PASSED\n");
}

// ===========================================================================
// 3. MOVE LEGALITY INVARIANT — no move should leave king in check
// ===========================================================================

fn run_legality_check() {
    println!("=== MOVE LEGALITY INVARIANT CHECK ===\n");

    let mut rng = rand::rng();
    let mut positions_checked = 0u64;
    let mut moves_checked = 0u64;

    // Play many random games and verify every legal move is truly legal
    for game_num in 0..100 {
        let mut state = GameState::new();
        for _ in 0..150 {
            if state.is_game_over() { break; }

            let legal_moves = state.legal_moves();
            for mv in &legal_moves {
                // Apply the move and check king is not in check
                state.apply_move(*mv);
                let stm_before = state.board.side_to_move.opponent(); // who just moved
                let in_check = movegen::is_in_check(&state.board, stm_before);
                state.undo_move();

                assert!(!in_check,
                    "Game {}: legal move {:?} leaves own king in check!",
                    game_num, mv);
                moves_checked += 1;
            }

            // Play a random move to advance
            let idx = rng.random_range(0..legal_moves.len());
            state.apply_move(legal_moves[idx]);
            positions_checked += 1;
        }
    }

    println!("  Checked {} positions, {} moves — all legal", positions_checked, moves_checked);
    println!("\n  LEGALITY INVARIANT PASSED\n");
}

// ===========================================================================
// 4. EN PASSANT EDGE CASES
// ===========================================================================

fn run_en_passant_tests() {
    println!("=== EN PASSANT EDGE CASE TESTS ===\n");

    let mut rng = rand::rng();
    let mut ep_captures_seen = 0;

    // Play random games, track en passant occurrences
    for _ in 0..500 {
        let mut state = GameState::new();
        for _ in 0..200 {
            if state.is_game_over() { break; }
            let moves = state.legal_moves();
            if moves.is_empty() { break; }

            // Count EP moves
            let ep_moves: Vec<_> = moves.iter().filter(|m| m.is_en_passant).collect();
            if !ep_moves.is_empty() {
                ep_captures_seen += ep_moves.len();

                // Verify EP capture removes the right pawn
                for ep_mv in &ep_moves {
                    let side = state.side_to_move();
                    state.apply_move(**ep_mv);

                    // The captured pawn's square should now be empty
                    let captured_sq = match side {
                        Color::White => HexCoord::new(ep_mv.to.q, ep_mv.to.r - 1),
                        Color::Black => HexCoord::new(ep_mv.to.q, ep_mv.to.r + 1),
                    };
                    assert!(state.board.get(captured_sq).is_none(),
                        "EP capture should remove pawn at {:?}", captured_sq);

                    state.undo_move();
                }
            }

            let idx = rng.random_range(0..moves.len());
            state.apply_move(moves[idx]);
        }
    }

    println!("  Saw {} en passant captures across 500 games", ep_captures_seen);
    assert!(ep_captures_seen > 0, "Should see at least some EP captures in 500 games");
    println!("\n  EN PASSANT PASSED\n");
}

// ===========================================================================
// 5. HEURISTIC VALUE FUNCTION — does it prefer winning positions?
// ===========================================================================

fn run_value_function_tests() {
    println!("=== VALUE FUNCTION TESTS ===\n");

    use hexchess_engine::board::PieceKind;
    use hexchess_engine::eval;

    // Test 1: Starting position should be 0
    let state = GameState::new();
    let score = eval::evaluate(&state);
    println!("  Starting position eval: {} cp (should be 0)", score);
    assert_eq!(score, 0);

    // Test 2: Remove a black queen — White should be winning
    let mut state2 = GameState::new();
    // Find and remove black queen
    for idx in 0..91 {
        if let Some(p) = state2.board.cells[idx] {
            if p.kind == PieceKind::Queen && p.color == Color::Black {
                let coord = hexchess_engine::board::index_to_coord(idx);
                state2.board.set(coord, None);
                break;
            }
        }
    }
    let score2 = eval::evaluate(&state2);
    println!("  White missing nothing, Black missing queen: {} cp (should be +900)", score2);
    assert_eq!(score2, 900);

    // Test 3: HeuristicEvaluator value for same position
    let eval = HeuristicEvaluator;
    let (_, value) = eval.evaluate(&state2);
    println!("  Heuristic value for +900cp position: {:.3} (should be positive)", value);
    assert!(value > 0.5, "value should be strongly positive for +900cp, got {}", value);

    // Test 4: Symmetric — give Black an extra queen instead
    let mut state3 = GameState::new();
    for idx in 0..91 {
        if let Some(p) = state3.board.cells[idx] {
            if p.kind == PieceKind::Queen && p.color == Color::White {
                let coord = hexchess_engine::board::index_to_coord(idx);
                state3.board.set(coord, None);
                break;
            }
        }
    }
    let score3 = eval::evaluate(&state3);
    println!("  White missing queen, Black full: {} cp (should be -900)", score3);
    assert_eq!(score3, -900);

    let (_, value3) = eval.evaluate(&state3);
    println!("  Heuristic value for -900cp position: {:.3} (should be negative)", value3);
    assert!(value3 < -0.5, "value should be strongly negative, got {}", value3);

    println!("\n  VALUE FUNCTION PASSED\n");
}

// ===========================================================================
// 6. MCTS TACTICAL TESTS — can MCTS find captures?
// ===========================================================================

fn run_mcts_tactical_tests() {
    println!("=== MCTS TACTICAL TESTS ===\n");

    use hexchess_engine::board::{Board, Piece, PieceKind};

    // Test 1: Free queen capture with rook adjacent
    // Simple: White rook next to undefended black queen
    let mut board = Board::empty();
    board.set(HexCoord::new(-5, 0), Some(Piece::new(PieceKind::King, Color::White)));
    board.white_king = HexCoord::new(-5, 0);
    board.set(HexCoord::new(5, 0), Some(Piece::new(PieceKind::King, Color::Black)));
    board.black_king = HexCoord::new(5, 0);
    // White rook one step from black queen (cardinal direction)
    board.set(HexCoord::new(0, 0), Some(Piece::new(PieceKind::Rook, Color::White)));
    board.set(HexCoord::new(0, 1), Some(Piece::new(PieceKind::Queen, Color::Black)));

    let state = GameState::from_board(board);
    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
    let result = search.search(&state, 500);

    let captures_queen = result.best_move.to == HexCoord::new(0, 1)
        && result.best_move.from == HexCoord::new(0, 0);
    println!("  Free queen capture: {} -> {} (should be (0,0)->(0,1)): {}",
        result.best_move.from, result.best_move.to,
        if captures_queen { "FOUND" } else { "MISSED" });
    assert!(captures_queen, "MCTS should find adjacent free queen capture");

    // Test 2: Verify value is positive after capturing
    // (Just check the search value estimate is positive — White is up material)
    println!("  Search value after capture position: {:.3} (should be positive)", result.value);
    assert!(result.value > 0.0, "Value should be positive when we can capture a queen");

    println!("\n  MCTS TACTICAL TESTS PASSED\n");
}

// ===========================================================================
// 7. HEURISTIC MCTS VS RANDOM MOVES — material accumulation
// ===========================================================================

fn run_heuristic_vs_random_moves() {
    println!("=== HEURISTIC MCTS VS RANDOM MOVES ===\n");

    use hexchess_engine::eval;

    let num_games = 20;
    let sims = 100;
    let mut rng = rand::rng();

    let mut heuristic_material_lead = 0i64;
    let mut games_where_heuristic_leads = 0;

    // Heuristic (White) vs Random moves (Black) — no MCTS for Black
    println!("  Playing {} games: MCTS+Heuristic(W) vs Random(B)...", num_games);

    for i in 0..num_games {
        let mut state = GameState::new();
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));

        for _ in 0..100 { // 100 full moves = 200 ply
            if state.is_game_over() { break; }

            match state.side_to_move() {
                Color::White => {
                    let result = search.search(&state, sims);
                    state.apply_move(result.best_move);
                }
                Color::Black => {
                    let moves = state.legal_moves();
                    if moves.is_empty() { break; }
                    let idx = rng.random_range(0..moves.len());
                    state.apply_move(moves[idx]);
                }
            }
        }

        // Evaluate final material balance (from White's perspective)
        let white_mat = eval::material(&state.board, Color::White);
        let black_mat = eval::material(&state.board, Color::Black);
        let lead = (white_mat - black_mat) as i64;
        heuristic_material_lead += lead;
        if lead > 0 { games_where_heuristic_leads += 1; }

        let status_char = match state.status() {
            GameStatus::Checkmate(Color::White) => 'W',
            GameStatus::Checkmate(Color::Black) => 'L',
            GameStatus::Ongoing => '.',
            _ => 'D',
        };
        print!("{}", status_char);
        if (i + 1) % 10 == 0 { println!(); }
    }
    println!();

    let avg_lead = heuristic_material_lead as f64 / num_games as f64;
    println!("\n  Avg material lead (heuristic): {:.0} cp", avg_lead);
    println!("  Games where heuristic leads: {}/{}", games_where_heuristic_leads, num_games);

    assert!(avg_lead > 0.0,
        "Heuristic MCTS should accumulate material vs random, avg lead: {:.0}", avg_lead);
    assert!(games_where_heuristic_leads > num_games / 2,
        "Heuristic should lead in most games: {}/{}", games_where_heuristic_leads, num_games);

    // Now test with colors swapped
    let mut heuristic_material_lead_b = 0i64;
    let mut games_where_heuristic_leads_b = 0;

    println!("\n  Playing {} games: Random(W) vs MCTS+Heuristic(B)...", num_games);

    for i in 0..num_games {
        let mut state = GameState::new();
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));

        for _ in 0..100 {
            if state.is_game_over() { break; }

            match state.side_to_move() {
                Color::White => {
                    let moves = state.legal_moves();
                    if moves.is_empty() { break; }
                    let idx = rng.random_range(0..moves.len());
                    state.apply_move(moves[idx]);
                }
                Color::Black => {
                    let result = search.search(&state, sims);
                    state.apply_move(result.best_move);
                }
            }
        }

        let white_mat = eval::material(&state.board, Color::White);
        let black_mat = eval::material(&state.board, Color::Black);
        let lead = (black_mat - white_mat) as i64; // heuristic is Black
        heuristic_material_lead_b += lead;
        if lead > 0 { games_where_heuristic_leads_b += 1; }

        let status_char = match state.status() {
            GameStatus::Checkmate(Color::Black) => 'W',
            GameStatus::Checkmate(Color::White) => 'L',
            GameStatus::Ongoing => '.',
            _ => 'D',
        };
        print!("{}", status_char);
        if (i + 1) % 10 == 0 { println!(); }
    }
    println!();

    let avg_lead_b = heuristic_material_lead_b as f64 / num_games as f64;
    println!("\n  Avg material lead (heuristic as Black): {:.0} cp", avg_lead_b);
    println!("  Games where heuristic leads: {}/{}", games_where_heuristic_leads_b, num_games);

    let total_leads = games_where_heuristic_leads + games_where_heuristic_leads_b;
    let total_avg = (heuristic_material_lead + heuristic_material_lead_b) as f64 / (num_games * 2) as f64;
    println!("\n  === OVERALL ===");
    println!("    Avg material lead: {:.0} cp", total_avg);
    println!("    Material lead in {}/{} games", total_leads, num_games * 2);

    assert!(total_avg > 100.0,
        "Heuristic should have >100cp avg lead vs random, got {:.0}", total_avg);

    println!("\n  HEURISTIC VS RANDOM PASSED\n");
}

// ===========================================================================
// 8. MOVE TABLE COMPLETENESS — every legal move must have an index
// ===========================================================================

fn run_move_table_completeness() {
    println!("=== MOVE TABLE COMPLETENESS CHECK ===\n");

    let mut rng = rand::rng();
    let mut total_moves_checked = 0u64;
    let mut unindexed = 0u64;

    // Play many random games and verify every legal move has a table entry
    for _ in 0..200 {
        let mut state = GameState::new();
        for _ in 0..200 {
            if state.is_game_over() { break; }
            let moves = state.legal_moves();
            for mv in &moves {
                if serialization::move_to_index(mv).is_none() {
                    eprintln!("  UNINDEXED MOVE: {:?}", mv);
                    unindexed += 1;
                }
                total_moves_checked += 1;
            }
            let idx = rng.random_range(0..moves.len());
            state.apply_move(moves[idx]);
        }
    }

    println!("  Checked {} legal moves across 200 random games", total_moves_checked);
    assert_eq!(unindexed, 0,
        "Found {} unindexed legal moves — move table is incomplete!", unindexed);

    // Also verify round-trip: index_to_move -> move_to_index for every index
    let n = serialization::num_move_indices();
    for i in 0..n {
        let (from, to, promo) = serialization::index_to_move(i);
        let mv = movegen::Move {
            from, to,
            promotion: promo,
            captured: None,
            is_en_passant: false,
        };
        let j = serialization::move_to_index(&mv);
        assert_eq!(j, Some(i), "Round-trip failed for index {}: got {:?}", i, j);
    }
    println!("  All {} move indices round-trip correctly", n);

    println!("\n  MOVE TABLE COMPLETENESS PASSED\n");
}

// ===========================================================================
// 9. MCTS WINS FROM WINNING POSITIONS
// ===========================================================================

fn run_mcts_winning_positions() {
    println!("=== MCTS WINNING POSITION TESTS ===\n");

    let sims = 500;

    // Test 1: Massive material advantage (queen + rook vs lone king)
    {
        let mut board = Board::empty();
        board.set(HexCoord::new(0, 0), Some(Piece::new(PieceKind::King, Color::White)));
        board.white_king = HexCoord::new(0, 0);
        board.set(HexCoord::new(0, 2), Some(Piece::new(PieceKind::Queen, Color::White)));
        board.set(HexCoord::new(2, -2), Some(Piece::new(PieceKind::Rook, Color::White)));
        board.set(HexCoord::new(5, -5), Some(Piece::new(PieceKind::King, Color::Black)));
        board.black_king = HexCoord::new(5, -5);

        let state = GameState::from_board(board);
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        let result = search.search(&state, sims);
        println!("  Q+R vs K value: {:.3} (should be strongly positive)", result.value);
        assert!(result.value > 0.3,
            "Q+R vs lone king should be very winning, got {:.3}", result.value);
    }

    // Test 2: Losing position — Black to move with lone king vs Q+R
    {
        let mut board = Board::empty();
        board.side_to_move = Color::Black;
        board.set(HexCoord::new(5, -5), Some(Piece::new(PieceKind::King, Color::Black)));
        board.black_king = HexCoord::new(5, -5);
        board.set(HexCoord::new(0, 0), Some(Piece::new(PieceKind::King, Color::White)));
        board.white_king = HexCoord::new(0, 0);
        board.set(HexCoord::new(0, 2), Some(Piece::new(PieceKind::Queen, Color::White)));
        board.set(HexCoord::new(2, -2), Some(Piece::new(PieceKind::Rook, Color::White)));

        let state = GameState::from_board(board);
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        let result = search.search(&state, sims);
        println!("  Lone K vs Q+R (Black to move) value: {:.3} (should be negative)", result.value);
        assert!(result.value < -0.3,
            "Lone king vs Q+R should be losing, got {:.3}", result.value);
    }

    // Test 3: Equal position should have value near zero
    {
        let state = GameState::new();
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        let result = search.search(&state, sims);
        println!("  Starting position value: {:.3} (should be near 0)", result.value);
        assert!(result.value.abs() < 0.3,
            "Starting position should be roughly equal, got {:.3}", result.value);
    }

    println!("\n  MCTS WINNING POSITIONS PASSED\n");
}

// ===========================================================================
// 10. MCTS FINDS FORCED TACTICS
// ===========================================================================

fn run_mcts_forced_tactics() {
    println!("=== MCTS FORCED TACTICS TESTS ===\n");

    let sims = 800;

    // Test 1: Queen under attack by rook — must move the queen
    {
        let mut board = Board::empty();
        board.set(HexCoord::new(-5, 5), Some(Piece::new(PieceKind::King, Color::White)));
        board.white_king = HexCoord::new(-5, 5);
        board.set(HexCoord::new(5, -5), Some(Piece::new(PieceKind::King, Color::Black)));
        board.black_king = HexCoord::new(5, -5);
        // White queen on (0,0), black rook on (0,3) attacking along the file
        board.set(HexCoord::new(0, 0), Some(Piece::new(PieceKind::Queen, Color::White)));
        board.set(HexCoord::new(0, 3), Some(Piece::new(PieceKind::Rook, Color::Black)));

        let state = GameState::from_board(board);
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        let result = search.search(&state, sims);

        let moves_queen = result.best_move.from == HexCoord::new(0, 0);
        let captures_rook = result.best_move.to == HexCoord::new(0, 3);
        println!("  Queen under attack: {} -> {} (moves queen: {}, captures rook: {})",
            result.best_move.from, result.best_move.to, moves_queen, captures_rook);
        assert!(moves_queen,
            "MCTS should move the attacked queen, instead moved from {}", result.best_move.from);
    }

    // Test 2: Capture a free piece — rook takes undefended bishop
    {
        let mut board = Board::empty();
        board.set(HexCoord::new(-5, 0), Some(Piece::new(PieceKind::King, Color::White)));
        board.white_king = HexCoord::new(-5, 0);
        board.set(HexCoord::new(5, 0), Some(Piece::new(PieceKind::King, Color::Black)));
        board.black_king = HexCoord::new(5, 0);
        board.set(HexCoord::new(0, 0), Some(Piece::new(PieceKind::Rook, Color::White)));
        board.set(HexCoord::new(0, 3), Some(Piece::new(PieceKind::Bishop, Color::Black)));

        let state = GameState::from_board(board);
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        let result = search.search(&state, sims);

        let captures = result.best_move.from == HexCoord::new(0, 0)
            && result.best_move.to == HexCoord::new(0, 3);
        println!("  Free bishop capture: {} -> {} (captured: {})",
            result.best_move.from, result.best_move.to, captures);
        assert!(captures, "MCTS should capture the free bishop");
    }

    println!("\n  MCTS FORCED TACTICS PASSED\n");
}

// ===========================================================================
// 11. MORE SIMULATIONS => BETTER PLAY
// ===========================================================================

fn run_simulation_scaling() {
    println!("=== SIMULATION SCALING TEST ===\n");

    use hexchess_engine::eval;

    let low_sims = 30;
    let high_sims = 200;
    let games_per_side = 10;

    let mut high_material_lead = 0i64;
    let mut games_where_high_leads = 0;
    let total_games = games_per_side * 2;

    // High sims plays White
    print!("  High(W) vs Low(B): ");
    for _ in 0..games_per_side {
        let mut state = GameState::new();
        let mut search_high = MctsSearch::new(Box::new(HeuristicEvaluator));
        let mut search_low = MctsSearch::new(Box::new(HeuristicEvaluator));

        for _ in 0..100 {
            if state.is_game_over() { break; }
            match state.side_to_move() {
                Color::White => {
                    let r = search_high.search(&state, high_sims);
                    state.apply_move(r.best_move);
                }
                Color::Black => {
                    let r = search_low.search(&state, low_sims);
                    state.apply_move(r.best_move);
                }
            }
        }
        let lead = (eval::material(&state.board, Color::White)
            - eval::material(&state.board, Color::Black)) as i64;
        high_material_lead += lead;
        if lead > 0 { games_where_high_leads += 1; }
        print!(".");
    }
    println!();

    // High sims plays Black
    print!("  Low(W) vs High(B): ");
    for _ in 0..games_per_side {
        let mut state = GameState::new();
        let mut search_high = MctsSearch::new(Box::new(HeuristicEvaluator));
        let mut search_low = MctsSearch::new(Box::new(HeuristicEvaluator));

        for _ in 0..100 {
            if state.is_game_over() { break; }
            match state.side_to_move() {
                Color::White => {
                    let r = search_low.search(&state, low_sims);
                    state.apply_move(r.best_move);
                }
                Color::Black => {
                    let r = search_high.search(&state, high_sims);
                    state.apply_move(r.best_move);
                }
            }
        }
        // Lead from Black's (high-sim) perspective
        let lead = (eval::material(&state.board, Color::Black)
            - eval::material(&state.board, Color::White)) as i64;
        high_material_lead += lead;
        if lead > 0 { games_where_high_leads += 1; }
        print!(".");
    }
    println!();

    let avg_lead = high_material_lead as f64 / total_games as f64;
    println!("\n  {}sims vs {}sims over {} games:", high_sims, low_sims, total_games);
    println!("  Avg material lead (high-sim player): {:.0} cp", avg_lead);
    println!("  Games where high-sim leads: {}/{}", games_where_high_leads, total_games);

    assert!(avg_lead > 0.0,
        "Higher simulations should accumulate material advantage, got {:.0} cp", avg_lead);

    println!("\n  SIMULATION SCALING PASSED\n");
}

// ===========================================================================
// 12. MCTS VALUE MONOTONICITY — more material => higher value
// ===========================================================================

fn run_value_monotonicity() {
    println!("=== VALUE MONOTONICITY TEST ===\n");

    let sims = 300;

    // Position A: White has K+Q+R vs lone K
    let mut board_a = Board::empty();
    board_a.set(HexCoord::new(-5, 5), Some(Piece::new(PieceKind::King, Color::White)));
    board_a.white_king = HexCoord::new(-5, 5);
    board_a.set(HexCoord::new(0, 0), Some(Piece::new(PieceKind::Queen, Color::White)));
    board_a.set(HexCoord::new(2, -2), Some(Piece::new(PieceKind::Rook, Color::White)));
    board_a.set(HexCoord::new(5, -5), Some(Piece::new(PieceKind::King, Color::Black)));
    board_a.black_king = HexCoord::new(5, -5);

    // Position B: White has K+Q vs lone K (less material)
    let mut board_b = Board::empty();
    board_b.set(HexCoord::new(-5, 5), Some(Piece::new(PieceKind::King, Color::White)));
    board_b.white_king = HexCoord::new(-5, 5);
    board_b.set(HexCoord::new(0, 0), Some(Piece::new(PieceKind::Queen, Color::White)));
    board_b.set(HexCoord::new(5, -5), Some(Piece::new(PieceKind::King, Color::Black)));
    board_b.black_king = HexCoord::new(5, -5);

    // Position C: K+N vs K (smaller advantage than K+Q)
    let mut board_c = Board::empty();
    board_c.set(HexCoord::new(-5, 5), Some(Piece::new(PieceKind::King, Color::White)));
    board_c.white_king = HexCoord::new(-5, 5);
    board_c.set(HexCoord::new(0, 0), Some(Piece::new(PieceKind::Knight, Color::White)));
    board_c.set(HexCoord::new(5, -5), Some(Piece::new(PieceKind::King, Color::Black)));
    board_c.black_king = HexCoord::new(5, -5);

    let state_a = GameState::from_board(board_a);
    let state_b = GameState::from_board(board_b);
    let state_c = GameState::from_board(board_c);

    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
    let va = search.search(&state_a, sims).value;
    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
    let vb = search.search(&state_b, sims).value;
    let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
    let vc = search.search(&state_c, sims).value;

    println!("  K+Q+R vs K: {:.3}", va);
    println!("  K+Q   vs K: {:.3}", vb);
    println!("  K+N   vs K: {:.3}", vc);

    assert!(va > vb, "More material should give higher value: {:.3} vs {:.3}", va, vb);
    assert!(vb > vc, "Queen advantage should beat knight advantage: {:.3} vs {:.3}", vb, vc);
    assert!(va > 0.0, "K+Q+R vs K should be positive");
    assert!(vb > 0.0, "K+Q vs K should be positive");
    assert!(vc > 0.0, "K+N vs K should be positive");

    println!("\n  VALUE MONOTONICITY PASSED\n");
}

// ===========================================================================
// Main
// ===========================================================================

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║     HEXCHESS ENGINE VALIDATION SUITE         ║");
    println!("╚══════════════════════════════════════════════╝\n");

    run_perft_tests();
    run_apply_undo_tests();
    run_legality_check();
    run_en_passant_tests();
    run_value_function_tests();
    run_mcts_tactical_tests();
    run_heuristic_vs_random_moves();
    run_move_table_completeness();
    run_mcts_winning_positions();
    run_mcts_forced_tactics();
    run_simulation_scaling();
    run_value_monotonicity();

    println!("╔══════════════════════════════════════════════╗");
    println!("║     ALL VALIDATION TESTS PASSED              ║");
    println!("╚══════════════════════════════════════════════╝");
}
