//! Encoder and policy-index mirror-equivariance tests.
//!
//! These tests verify the core STM-frame symmetry invariants across 100
//! diverse positions from the stratified position generator, catching
//! the class of bugs fixed in #109 at the full-tensor and policy-index levels.

mod helpers;

use hexchess_engine::board::{ALL_COORDS, Color, Piece};
use hexchess_engine::game::GameState;
use hexchess_engine::movegen;
use hexchess_engine::serialization::{
    BOARD_DIM, TENSOR_SIZE, encode_board, mirror_coord, stm_policy_index,
};

use helpers::positions::{PlyBucket, generate_positions};

// =========================================================================
// Helpers
// =========================================================================

/// Build the color-mirrored game state: every piece's color is swapped,
/// every coordinate is centrally inverted, and the side-to-move is flipped.
/// This is the "same game from the other player's seat" transformation.
fn mirror_game_state(state: &GameState) -> GameState {
    let board = &state.board;
    let mut mirrored = hexchess_engine::board::Board::empty();
    mirrored.side_to_move = board.side_to_move.opponent();
    mirrored.halfmove_clock = board.halfmove_clock;
    mirrored.fullmove_number = board.fullmove_number;
    mirrored.en_passant = board.en_passant.map(mirror_coord);

    for &coord in ALL_COORDS.iter() {
        if let Some(piece) = board.get(coord) {
            let mc = mirror_coord(coord);
            mirrored.set(mc, Some(Piece::new(piece.kind, piece.color.opponent())));
            if piece.kind == hexchess_engine::board::PieceKind::King {
                if piece.color == Color::White {
                    // White king becomes black king in the mirrored board
                    mirrored.black_king = mc;
                } else {
                    // Black king becomes white king in the mirrored board
                    mirrored.white_king = mc;
                }
            }
        }
    }

    GameState::from_board(mirrored)
}

/// Extract a single channel plane from a flat CHW tensor.
fn plane(tensor: &[f32; TENSOR_SIZE], ch: usize) -> Vec<f32> {
    let start = ch * BOARD_DIM * BOARD_DIM;
    tensor[start..start + BOARD_DIM * BOARD_DIM].to_vec()
}

// =========================================================================
// 1. Encoder mirror equivariance (full tensor level)
// =========================================================================

/// For diverse positions from the generator, verify that encoding the
/// color-mirrored position produces the same tensor for piece channels.
///
/// The STM frame is designed so that `encode(S)` and `encode(mirror(S))`
/// produce identical piece representations: the mirror transformation
/// (invert coords + swap colors + flip STM) is exactly cancelled by the
/// encoder's STM-frame normalization (which inverts coords and swaps
/// channels for black-to-move). This is the core invariant that makes
/// the neural network color-agnostic.
///
/// Channels tested:
///   - 0-11 (pieces): must be identical
///   - 13, 14, 21 (fullmove, halfmove, plies-since-pawn): must be identical
///   - 15 (en passant): must be identical (STM frame mirrors EP target too)
///   - 17 (validity mask): must be identical
///   - 12 (absolute STM): must flip (1→0 or 0→1)
///
/// Channel 18 (in-check) is board-state-derived and must also be identical.
///
/// Channels 16 (repetition) and 19-20 (last move) are skipped because
/// `mirror_game_state` constructs a fresh GameState without move history,
/// so these channels are always zero in the mirrored encoding.
#[test]
fn encoder_mirror_equivariance_100_positions() {
    let mut all_positions = Vec::new();
    all_positions.extend(generate_positions(1000, PlyBucket::Opening, 34));
    all_positions.extend(generate_positions(2000, PlyBucket::Midgame, 33));
    all_positions.extend(generate_positions(3000, PlyBucket::Endgame, 33));

    assert!(
        all_positions.len() >= 98,
        "need at least 98 positions, got {}",
        all_positions.len()
    );

    for (i, state) in all_positions.iter().enumerate() {
        let t_orig = encode_board(state);
        let mirrored = mirror_game_state(state);
        let t_mirror = encode_board(&mirrored);

        // Piece planes (0-11): encoding the mirrored position through the
        // STM frame should produce identical piece channels, because the
        // mirror (invert + color-swap) and STM-frame (invert + channel-swap)
        // transformations cancel each other out.
        for ch in 0..12 {
            let orig_p = plane(&t_orig, ch);
            let mirror_p = plane(&t_mirror, ch);
            assert_eq!(
                orig_p, mirror_p,
                "position {i}: piece plane ch {ch} differs after mirror"
            );
        }

        // Validity mask (ch 17): must be identical (both positions have the
        // same 91 valid cells, just viewed from opposite STM frames).
        assert_eq!(
            plane(&t_orig, 17),
            plane(&t_mirror, 17),
            "position {i}: validity mask differs"
        );

        // En passant (ch 15): must be identical (STM frame mirrors EP too).
        assert_eq!(
            plane(&t_orig, 15),
            plane(&t_mirror, 15),
            "position {i}: en passant plane differs"
        );

        // In-check plane (ch 18): board-state-derived, must be identical.
        assert_eq!(
            plane(&t_orig, 18),
            plane(&t_mirror, 18),
            "position {i}: in-check plane differs"
        );

        // Scalar constant planes (13=fullmove, 14=halfmove, 21=plies-since-pawn).
        for ch in [13, 14, 21] {
            assert_eq!(
                plane(&t_orig, ch),
                plane(&t_mirror, ch),
                "position {i}: scalar plane ch {ch} differs"
            );
        }

        // Absolute STM plane (ch 12) should flip: 1.0 → 0.0 or vice versa.
        let orig_stm_val = t_orig[12 * BOARD_DIM * BOARD_DIM];
        let mirror_stm_val = t_mirror[12 * BOARD_DIM * BOARD_DIM];
        assert!(
            (orig_stm_val - (1.0 - mirror_stm_val)).abs() < 1e-6,
            "position {i}: STM plane should flip"
        );
    }
}

// =========================================================================
// 2. Policy-index mirror equivariance
// =========================================================================

/// For every legal move in 100 diverse positions, verify that the STM
/// policy index is equivariant under the color-mirror transformation:
///   stm_policy_index(mv, stm) == stm_policy_index(mirror(mv), !stm)
///
/// This targets the exact class of bug fixed in #109 where policy indices
/// were not correctly remapped for black-to-move positions.
#[test]
fn policy_index_mirror_equivariance_100_positions() {
    let mut all_positions = Vec::new();
    all_positions.extend(generate_positions(4000, PlyBucket::Opening, 34));
    all_positions.extend(generate_positions(5000, PlyBucket::Midgame, 33));
    all_positions.extend(generate_positions(6000, PlyBucket::Endgame, 33));

    assert!(
        all_positions.len() >= 98,
        "need at least 98 positions, got {}",
        all_positions.len()
    );

    let mut total_moves = 0usize;

    for (i, state) in all_positions.iter().enumerate() {
        let stm = state.side_to_move();
        let moves = state.legal_moves();

        for mv in &moves {
            let idx = stm_policy_index(mv, stm)
                .unwrap_or_else(|| panic!("position {i}: move {mv:?} has no policy index"));

            // Mirror the move: (from, to) → (mirror(from), mirror(to)),
            // same promotion piece.
            let mirror_mv = movegen::Move {
                from: mirror_coord(mv.from),
                to: mirror_coord(mv.to),
                promotion: mv.promotion,
                captured: None,
                is_en_passant: false,
            };

            let mirror_idx = stm_policy_index(&mirror_mv, stm.opponent()).unwrap_or_else(|| {
                panic!("position {i}: mirrored move {mirror_mv:?} has no policy index")
            });

            assert_eq!(
                idx, mirror_idx,
                "position {i}: policy index mismatch for move {mv:?} (idx={idx}) \
                 vs mirrored {mirror_mv:?} (idx={mirror_idx})"
            );

            total_moves += 1;
        }
    }

    // Sanity: we should have tested thousands of moves.
    assert!(
        total_moves > 1000,
        "only tested {total_moves} moves — something is wrong with position generation"
    );
}
