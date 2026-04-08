//! Board-to-tensor encoding, move-to-index bijection, and training record
//! serialization for the hexagonal chess neural network pipeline.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use crate::board::{
    CARDINAL_DIRS, Color, DIAGONAL_DIRS, HexCoord, KNIGHT_OFFSETS, PROMOTION_PIECES, PieceKind,
};
use crate::game::GameState;
use crate::movegen::{self, Move};

// ---------------------------------------------------------------------------
// Board-to-tensor encoding
// ---------------------------------------------------------------------------

pub const NUM_CHANNELS: usize = 22;
pub const BOARD_DIM: usize = 11;
pub const TENSOR_SIZE: usize = NUM_CHANNELS * BOARD_DIM * BOARD_DIM;

/// Encode a game state as a flat f32 tensor in CHW layout.
///
/// The 91-cell hex grid is embedded into an 11x11 rectangular array using
/// `col = q + 5`, `row = r + 5`. Invalid cells are zero-padded.
///
/// Channels:
///  0: White Pawn    1: White Knight   2: White Bishop
///  3: White Rook    4: White Queen    5: White King
///  6: Black Pawn    7: Black Knight   8: Black Bishop
///  9: Black Rook   10: Black Queen   11: Black King
/// 12: Side to move (1.0 = white, 0.0 = black; constant plane)
/// 13: Move count (fullmove_number / 100.0; constant plane)
/// 14: Halfmove clock (halfmove_clock / 100.0; constant plane)
/// 15: En passant target (1.0 on the target cell, 0.0 elsewhere)
/// 16: Repetition count (0.0 / 1.0 / 2.0; constant plane)
/// 17: Validity mask (1.0 on 91 valid hex cells, 0.0 on padding)
/// 18: In check (1.0 if side to move is in check; constant plane)
/// 19: Last move from-square (1.0 at source of most recent move, else 0)
/// 20: Last move to-square (1.0 at destination of most recent move, else 0)
/// 21: Plies-since-pawn-move (halfmove_clock / 50, clamped to [0, 1]; constant
///     plane). NOTE: we reuse the board's halfmove clock, which in this engine
///     resets on pawn moves OR captures (standard halfmove-clock semantics).
///     The plan specifies "plies since pawn move" but we deliberately reuse
///     halfmove_clock here rather than adding new state — the two signals are
///     highly correlated and the net can learn the difference if it matters.
pub fn encode_board(state: &GameState) -> [f32; TENSOR_SIZE] {
    let board = &state.board;
    let mut tensor = [0.0f32; TENSOR_SIZE];

    // Helper: index into the flat CHW tensor.
    let idx = |channel: usize, col: usize, row: usize| -> usize {
        channel * BOARD_DIM * BOARD_DIM + row * BOARD_DIM + col
    };

    // Piece planes (channels 0-11) and validity mask (channel 17).
    for q in -5i8..=5 {
        for r in -5i8..=5 {
            let coord = HexCoord::new(q, r);
            if !coord.is_valid() {
                continue;
            }
            let col = (q + 5) as usize;
            let row = (r + 5) as usize;

            // Validity mask
            tensor[idx(17, col, row)] = 1.0;

            if let Some(piece) = board.get(coord) {
                let channel = piece_channel(piece.color, piece.kind);
                tensor[idx(channel, col, row)] = 1.0;
            }
        }
    }

    // Side to move (channel 12): constant plane.
    let side_val = match board.side_to_move {
        Color::White => 1.0f32,
        Color::Black => 0.0f32,
    };
    for row in 0..BOARD_DIM {
        for col in 0..BOARD_DIM {
            tensor[idx(12, col, row)] = side_val;
        }
    }

    // Move count (channel 13): constant plane.
    let move_val = board.fullmove_number as f32 / 100.0;
    for row in 0..BOARD_DIM {
        for col in 0..BOARD_DIM {
            tensor[idx(13, col, row)] = move_val;
        }
    }

    // Halfmove clock (channel 14): constant plane.
    let half_val = board.halfmove_clock as f32 / 100.0;
    for row in 0..BOARD_DIM {
        for col in 0..BOARD_DIM {
            tensor[idx(14, col, row)] = half_val;
        }
    }

    // En passant (channel 15): single cell.
    if let Some(ep) = board.en_passant
        && ep.is_valid()
    {
        let col = (ep.q + 5) as usize;
        let row = (ep.r + 5) as usize;
        tensor[idx(15, col, row)] = 1.0;
    }

    // Repetition count (channel 16): constant plane.
    let rep_val = state.repetition_count().min(2) as f32;
    if rep_val > 0.0 {
        for row in 0..BOARD_DIM {
            for col in 0..BOARD_DIM {
                tensor[idx(16, col, row)] = rep_val;
            }
        }
    }

    // In check (channel 18): constant plane.
    if movegen::is_in_check(board, board.side_to_move) {
        for row in 0..BOARD_DIM {
            for col in 0..BOARD_DIM {
                tensor[idx(18, col, row)] = 1.0;
            }
        }
    }

    // Last-move from/to (channels 19, 20).
    if let Some(mv) = state.last_move() {
        if mv.from.is_valid() {
            let col = (mv.from.q + 5) as usize;
            let row = (mv.from.r + 5) as usize;
            tensor[idx(19, col, row)] = 1.0;
        }
        if mv.to.is_valid() {
            let col = (mv.to.q + 5) as usize;
            let row = (mv.to.r + 5) as usize;
            tensor[idx(20, col, row)] = 1.0;
        }
    }

    // Plies-since-pawn-move (channel 21): constant plane, halfmove_clock / 50
    // clamped to [0, 1]. See the channel doc comment above for why we reuse
    // halfmove_clock directly.
    let pspm = (board.halfmove_clock as f32 / 50.0).clamp(0.0, 1.0);
    if pspm > 0.0 {
        for row in 0..BOARD_DIM {
            for col in 0..BOARD_DIM {
                tensor[idx(21, col, row)] = pspm;
            }
        }
    }

    tensor
}

/// Map (color, piece_kind) to a channel index 0..11.
fn piece_channel(color: Color, kind: PieceKind) -> usize {
    color.index() * 6 + kind.index()
}

// ---------------------------------------------------------------------------
// Move-to-index bijection
// ---------------------------------------------------------------------------

/// A canonical (from, to, promotion) tuple for the move index table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct MoveEntry {
    from_idx: u8,                 // 0..90  (cell index of source)
    to_idx: u8,                   // 0..90  (cell index of destination)
    promotion: Option<PieceKind>, // None for non-promotion moves
}

/// Maximum ray length on the board (diameter is 10, so max 10 steps).
const MAX_RAY_LEN: usize = 10;

/// Returns true if `coord` sits on a promotion rank for the given color.
///
/// In Glinski's variant:
/// - White promotes on the far edge (the cells with maximal s = -(q+r) = 5,
///   i.e. q + r == -5). But actually promotion happens on the opponent's back
///   rank. The back rank for black (where white promotes) is the set of cells
///   with r == 5 - max(0, q), essentially the top edge of the hex.
///
/// For white, promotion cells are those on the "top" edge:
///   r = min(5, 5 - q) when q >= 0, i.e. the cells where r is at its maximum
///   for that column.
///
/// For black, promotion cells are those on the "bottom" edge.
///
/// Specifically, a cell (q, r) is on white's promotion rank iff it's valid and
/// there is no valid cell at (q, r+1). Similarly for black at (q, r-1).
///
/// But for move-table generation we just need to know WHICH destination cells
/// are promotion squares for EITHER color. A pawn of either color that reaches
/// the far edge must promote.
fn is_promotion_cell_for_white(coord: HexCoord) -> bool {
    // White promotes on the cells at the maximum r for each column.
    // A valid cell (q, r) is on white's promotion rank if (q, r+1) is invalid.
    coord.is_valid() && !HexCoord::new(coord.q, coord.r + 1).is_valid()
}

fn is_promotion_cell_for_black(coord: HexCoord) -> bool {
    // Black promotes on the cells at the minimum r for each column.
    coord.is_valid() && !HexCoord::new(coord.q, coord.r - 1).is_valid()
}

/// Returns true if `coord` is a promotion cell for either color.
pub fn is_promotion_cell(coord: HexCoord) -> bool {
    is_promotion_cell_for_white(coord) || is_promotion_cell_for_black(coord)
}

/// Build the complete, sorted list of all move entries.
fn build_move_table() -> Vec<MoveEntry> {
    use crate::board::ALL_COORDS;

    // Collect all unique (from, to) pairs reachable by any piece.
    let mut pairs: BTreeSet<(u8, u8)> = BTreeSet::new();
    // Track which (from, to) pairs are pawn moves to promotion cells.
    // Only these get expanded to 4 promotion entries; other moves to
    // promotion cells keep a single non-promotion entry.
    let mut pawn_promo_pairs: BTreeSet<(u8, u8)> = BTreeSet::new();

    for (from_idx, &from_coord) in ALL_COORDS.iter().enumerate() {
        let fi = from_idx as u8;

        // --- Knight jumps ---
        for &(dq, dr) in &KNIGHT_OFFSETS {
            let to = HexCoord::new(from_coord.q + dq, from_coord.r + dr);
            if to.is_valid() {
                let ti = crate::board::coord_to_index(to).unwrap() as u8;
                pairs.insert((fi, ti));
            }
        }

        // --- Sliding rays (cardinal directions for rook/queen) ---
        for &(dq, dr) in &CARDINAL_DIRS {
            for dist in 1..=MAX_RAY_LEN {
                let to = HexCoord::new(
                    from_coord.q + dq * dist as i8,
                    from_coord.r + dr * dist as i8,
                );
                if !to.is_valid() {
                    break;
                }
                let ti = crate::board::coord_to_index(to).unwrap() as u8;
                pairs.insert((fi, ti));
            }
        }

        // --- Sliding rays (diagonal directions for bishop/queen) ---
        for &(dq, dr) in &DIAGONAL_DIRS {
            for dist in 1..=MAX_RAY_LEN {
                let to = HexCoord::new(
                    from_coord.q + dq * dist as i8,
                    from_coord.r + dr * dist as i8,
                );
                if !to.is_valid() {
                    break;
                }
                let ti = crate::board::coord_to_index(to).unwrap() as u8;
                pairs.insert((fi, ti));
            }
        }

        // --- Pawn moves (both colors from this cell) ---
        // White pawn forward: (0, 1), captures: (1, 0) and (-1, 1), double: (0, 2).
        let white_pawn_dests: [(i8, i8); 4] = [(0, 1), (0, 2), (1, 0), (-1, 1)];
        for &(dq, dr) in &white_pawn_dests {
            let to = HexCoord::new(from_coord.q + dq, from_coord.r + dr);
            if to.is_valid() {
                let ti = crate::board::coord_to_index(to).unwrap() as u8;
                pairs.insert((fi, ti));
                if is_promotion_cell_for_white(to) {
                    pawn_promo_pairs.insert((fi, ti));
                }
            }
        }

        // Black pawn forward: (0, -1), captures: (-1, 0) and (1, -1), double: (0, -2).
        let black_pawn_dests: [(i8, i8); 4] = [(0, -1), (0, -2), (-1, 0), (1, -1)];
        for &(dq, dr) in &black_pawn_dests {
            let to = HexCoord::new(from_coord.q + dq, from_coord.r + dr);
            if to.is_valid() {
                let ti = crate::board::coord_to_index(to).unwrap() as u8;
                pairs.insert((fi, ti));
                if is_promotion_cell_for_black(to) {
                    pawn_promo_pairs.insert((fi, ti));
                }
            }
        }
    }

    // Expand (from, to) pairs into MoveEntry values.
    // Pawn promotion pairs get 4 entries (one per promotion piece) instead of
    // the base non-promotion entry. All other pairs get a single non-promotion
    // entry, even if the destination is an edge cell (since non-pawn pieces
    // don't promote).
    let mut entries: Vec<MoveEntry> = Vec::new();

    for &(from_idx, to_idx) in &pairs {
        // Every pair gets a non-promotion entry (for non-pawn pieces that
        // can also reach promotion cells without promoting).
        entries.push(MoveEntry {
            from_idx,
            to_idx,
            promotion: None,
        });
        // Pawn promotion pairs additionally get 4 promotion entries.
        if pawn_promo_pairs.contains(&(from_idx, to_idx)) {
            for &pk in &PROMOTION_PIECES {
                entries.push(MoveEntry {
                    from_idx,
                    to_idx,
                    promotion: Some(pk),
                });
            }
        }
    }

    // Sort deterministically.
    entries.sort();
    entries
}

struct MoveIndex {
    /// Sorted list of all move entries; the index in this vec IS the move index.
    entries: Vec<MoveEntry>,
    /// Reverse lookup: (from_idx, to_idx, promotion_ordinal) -> move index.
    /// promotion_ordinal: 0=None, 1=Queen, 2=Rook, 3=Bishop, 4=Knight
    reverse: std::collections::HashMap<(u8, u8, u8), usize>,
}

fn promotion_ordinal(p: Option<PieceKind>) -> u8 {
    match p {
        None => 0,
        Some(PieceKind::Queen) => 1,
        Some(PieceKind::Rook) => 2,
        Some(PieceKind::Bishop) => 3,
        Some(PieceKind::Knight) => 4,
        Some(_) => unreachable!("only Q/R/B/N are valid promotion pieces"),
    }
}

static MOVE_INDEX: LazyLock<MoveIndex> = LazyLock::new(|| {
    let entries = build_move_table();
    let mut reverse = std::collections::HashMap::with_capacity(entries.len());
    for (i, e) in entries.iter().enumerate() {
        let key = (e.from_idx, e.to_idx, promotion_ordinal(e.promotion));
        reverse.insert(key, i);
    }
    MoveIndex { entries, reverse }
});

/// Total number of move indices in the policy vector.
///
/// This is computed once at startup and cached.
pub fn num_move_indices() -> usize {
    MOVE_INDEX.entries.len()
}

/// Convert a `Move` to a policy-vector index.
///
/// Returns `None` if the move is not in the table.
pub fn move_to_index(mv: &Move) -> Option<usize> {
    let fi = crate::board::coord_to_index(mv.from)? as u8;
    let ti = crate::board::coord_to_index(mv.to)? as u8;
    let po = promotion_ordinal(mv.promotion);
    MOVE_INDEX.reverse.get(&(fi, ti, po)).copied()
}

/// Convert a policy-vector index back to `(from, to, promotion)`.
///
/// Panics if `index >= num_move_indices()`.
pub fn index_to_move(index: usize) -> (HexCoord, HexCoord, Option<PieceKind>) {
    let e = &MOVE_INDEX.entries[index];
    let from = crate::board::index_to_coord(e.from_idx as usize);
    let to = crate::board::index_to_coord(e.to_idx as usize);
    (from, to, e.promotion)
}

// ---------------------------------------------------------------------------
// Training record serialization
// ---------------------------------------------------------------------------

pub struct TrainingRecord {
    pub board_tensor: [f32; TENSOR_SIZE],
    pub policy_target: Vec<f32>,
    /// WDL target: [win, draw, loss] probabilities.
    pub wdl_target: [f32; 3],
}

impl TrainingRecord {
    /// Size of one record in bytes.
    ///
    /// Layout: `[f32 x TENSOR_SIZE] [f32 x num_move_indices()] [f32 x 3]`
    pub fn record_size() -> usize {
        (TENSOR_SIZE + num_move_indices() + 3) * std::mem::size_of::<f32>()
    }

    /// Serialize to flat binary (f32 little-endian).
    pub fn to_bytes(&self) -> Vec<u8> {
        let n_move = num_move_indices();
        let total_floats = TENSOR_SIZE + n_move + 3;
        let mut buf = Vec::with_capacity(total_floats * 4);

        for &v in &self.board_tensor {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(
            self.policy_target.len(),
            n_move,
            "policy_target length mismatch"
        );
        for &v in &self.policy_target {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.wdl_target {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        buf
    }

    /// Deserialize from flat binary (f32 little-endian).
    ///
    /// Panics if `data.len() != Self::record_size()`.
    pub fn from_bytes(data: &[u8]) -> Self {
        let expected = Self::record_size();
        assert_eq!(
            data.len(),
            expected,
            "TrainingRecord::from_bytes: expected {} bytes, got {}",
            expected,
            data.len()
        );

        let n_move = num_move_indices();

        let mut offset = 0;

        let read_f32 = |off: &mut usize| -> f32 {
            let bytes: [u8; 4] = data[*off..*off + 4].try_into().unwrap();
            *off += 4;
            f32::from_le_bytes(bytes)
        };

        let mut board_tensor = [0.0f32; TENSOR_SIZE];
        for slot in board_tensor.iter_mut() {
            *slot = read_f32(&mut offset);
        }

        let mut policy_target = vec![0.0f32; n_move];
        for slot in policy_target.iter_mut() {
            *slot = read_f32(&mut offset);
        }

        let wdl_target = [
            read_f32(&mut offset),
            read_f32(&mut offset),
            read_f32(&mut offset),
        ];

        TrainingRecord {
            board_tensor,
            policy_target,
            wdl_target,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Color, HexCoord, Piece, PieceKind};
    use crate::game::GameState;
    use crate::movegen::Move;

    #[test]
    fn test_tensor_shape() {
        assert_eq!(NUM_CHANNELS, 22);
        assert_eq!(TENSOR_SIZE, 22 * 11 * 11);
        assert_eq!(TENSOR_SIZE, 2662);
    }

    fn plane_sum(tensor: &[f32; TENSOR_SIZE], ch: usize) -> f32 {
        let base = ch * BOARD_DIM * BOARD_DIM;
        tensor[base..base + BOARD_DIM * BOARD_DIM].iter().sum()
    }

    #[test]
    fn test_last_move_planes_zero_on_initial_position() {
        let state = GameState::new();
        let tensor = encode_board(&state);
        assert_eq!(plane_sum(&tensor, 19), 0.0);
        assert_eq!(plane_sum(&tensor, 20), 0.0);
        assert_eq!(plane_sum(&tensor, 21), 0.0);
    }

    #[test]
    fn test_last_move_planes_after_pawn_move() {
        let mut state = GameState::new();
        // Pick any legal pawn move from the starting position.
        let legal = state.legal_moves();
        // b1-b2 style: find a move whose source piece is a pawn.
        let pawn_move = legal
            .into_iter()
            .find(|m| {
                state
                    .board
                    .get(m.from)
                    .map(|p| p.kind == PieceKind::Pawn)
                    .unwrap_or(false)
            })
            .expect("starting position has pawn moves");
        let from = pawn_move.from;
        let to = pawn_move.to;
        state.apply_move(pawn_move);

        let tensor = encode_board(&state);
        let from_col = (from.q + 5) as usize;
        let from_row = (from.r + 5) as usize;
        let to_col = (to.q + 5) as usize;
        let to_row = (to.r + 5) as usize;
        assert_eq!(
            tensor[19 * BOARD_DIM * BOARD_DIM + from_row * BOARD_DIM + from_col],
            1.0,
            "last-move from-plane should mark source cell"
        );
        assert_eq!(
            tensor[20 * BOARD_DIM * BOARD_DIM + to_row * BOARD_DIM + to_col],
            1.0,
            "last-move to-plane should mark destination cell"
        );
        // Exactly one 1.0 in each plane.
        assert_eq!(plane_sum(&tensor, 19), 1.0);
        assert_eq!(plane_sum(&tensor, 20), 1.0);
        // Pawn move resets the halfmove clock.
        assert_eq!(plane_sum(&tensor, 21), 0.0);
    }

    #[test]
    fn test_plies_since_pawn_move_plane() {
        let mut board = Board::starting_position();
        board.halfmove_clock = 10;
        let state = GameState::from_board(board);
        let tensor = encode_board(&state);
        // 10 / 50 = 0.2 on every valid (and padding) cell.
        let val = tensor[21 * BOARD_DIM * BOARD_DIM];
        assert!((val - 0.2).abs() < 1e-6);
        // Sum over the full 11x11 plane.
        let expected = 0.2 * (BOARD_DIM * BOARD_DIM) as f32;
        assert!((plane_sum(&tensor, 21) - expected).abs() < 1e-4);
    }

    #[test]
    fn test_plies_since_pawn_move_plane_clamped() {
        let mut board = Board::starting_position();
        board.halfmove_clock = 200; // way past 50
        let state = GameState::from_board(board);
        let tensor = encode_board(&state);
        let val = tensor[21 * BOARD_DIM * BOARD_DIM];
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_encode_board_starting_position_piece_planes() {
        let state = GameState::new();
        let tensor = encode_board(&state);

        let mut piece_count = 0;
        for ch in 0..12 {
            for row in 0..BOARD_DIM {
                for col in 0..BOARD_DIM {
                    let val = tensor[ch * BOARD_DIM * BOARD_DIM + row * BOARD_DIM + col];
                    if val > 0.5 {
                        piece_count += 1;
                    }
                }
            }
        }
        // Glinski: 9 pawns + 2 rooks + 2 knights + 3 bishops + 1 queen + 1 king = 18 per side.
        assert_eq!(piece_count, 36, "expected 36 pieces in starting position");
    }

    #[test]
    fn test_encode_board_side_to_move_plane() {
        let state = GameState::new();
        let tensor = encode_board(&state);

        // Channel 12 should be all 1.0 (white to move).
        let ch = 12;
        for row in 0..BOARD_DIM {
            for col in 0..BOARD_DIM {
                let val = tensor[ch * BOARD_DIM * BOARD_DIM + row * BOARD_DIM + col];
                assert_eq!(val, 1.0, "side-to-move plane should be 1.0 for white");
            }
        }

        // Flip side to move.
        let mut board2 = Board::starting_position();
        board2.side_to_move = Color::Black;
        let state2 = GameState::from_board(board2);
        let tensor2 = encode_board(&state2);
        for row in 0..BOARD_DIM {
            for col in 0..BOARD_DIM {
                let val = tensor2[ch * BOARD_DIM * BOARD_DIM + row * BOARD_DIM + col];
                assert_eq!(val, 0.0, "side-to-move plane should be 0.0 for black");
            }
        }
    }

    #[test]
    fn test_encode_board_en_passant_plane() {
        let mut board = Board::starting_position();
        let ep = HexCoord::new(0, 0);
        board.en_passant = Some(ep);
        let state = GameState::from_board(board);
        let tensor = encode_board(&state);

        let ch = 15;
        let col = (ep.q + 5) as usize;
        let row = (ep.r + 5) as usize;
        assert_eq!(
            tensor[ch * BOARD_DIM * BOARD_DIM + row * BOARD_DIM + col],
            1.0,
            "en passant cell should be 1.0"
        );

        let mut count = 0;
        for r in 0..BOARD_DIM {
            for c in 0..BOARD_DIM {
                if tensor[ch * BOARD_DIM * BOARD_DIM + r * BOARD_DIM + c] > 0.5 {
                    count += 1;
                }
            }
        }
        assert_eq!(count, 1, "only one cell should be set in en passant plane");
    }

    #[test]
    fn test_encode_board_move_count_plane() {
        let mut board = Board::starting_position();
        board.fullmove_number = 50;
        board.halfmove_clock = 10;
        let state = GameState::from_board(board);
        let tensor = encode_board(&state);

        // Channel 13: fullmove / 100 = 0.5
        let val = tensor[13 * BOARD_DIM * BOARD_DIM]; // first cell
        assert!((val - 0.5).abs() < 1e-6);

        // Channel 14: halfmove / 100 = 0.1
        let val = tensor[14 * BOARD_DIM * BOARD_DIM];
        assert!((val - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_validity_mask_plane() {
        let state = GameState::new();
        let tensor = encode_board(&state);

        let ch = 17;
        let mut valid_count = 0;
        for row in 0..BOARD_DIM {
            for col in 0..BOARD_DIM {
                let val = tensor[ch * BOARD_DIM * BOARD_DIM + row * BOARD_DIM + col];
                if val > 0.5 {
                    valid_count += 1;
                }
            }
        }
        assert_eq!(valid_count, 91, "validity mask should have exactly 91 ones");
    }

    #[test]
    fn test_repetition_count_plane() {
        let state = GameState::new();
        let tensor = encode_board(&state);

        // Starting position: no repetitions, channel 16 should be all 0.
        let ch = 16;
        for row in 0..BOARD_DIM {
            for col in 0..BOARD_DIM {
                assert_eq!(
                    tensor[ch * BOARD_DIM * BOARD_DIM + row * BOARD_DIM + col],
                    0.0,
                    "repetition plane should be 0.0 for new game"
                );
            }
        }
    }

    #[test]
    fn test_repetition_count_plane_after_repeat() {
        // Set up a position where we can shuttle rooks to create a repetition.
        let mut board = Board::empty();
        board.set(
            HexCoord::new(0, -4),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        board.set(
            HexCoord::new(0, 4),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        board.set(
            HexCoord::new(2, -5),
            Some(Piece::new(PieceKind::Rook, Color::White)),
        );
        board.set(
            HexCoord::new(-2, 5),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        board.white_king = HexCoord::new(0, -4);
        board.black_king = HexCoord::new(0, 4);

        let mut state = GameState::from_board(board);

        // Shuttle rooks to repeat the starting position.
        state.apply_move(Move::new(HexCoord::new(2, -5), HexCoord::new(2, -3), None));
        state.apply_move(Move::new(HexCoord::new(-2, 5), HexCoord::new(-2, 3), None));
        state.apply_move(Move::new(HexCoord::new(2, -3), HexCoord::new(2, -5), None));
        state.apply_move(Move::new(HexCoord::new(-2, 3), HexCoord::new(-2, 5), None));
        // Now at repetition count 1.
        assert_eq!(state.repetition_count(), 1);

        let tensor = encode_board(&state);
        let ch = 16;
        // Check a valid cell has rep count 1.0
        let val = tensor[ch * BOARD_DIM * BOARD_DIM + 5 * BOARD_DIM + 5]; // center cell
        assert_eq!(val, 1.0, "repetition plane should be 1.0 after one repeat");
    }

    #[test]
    fn test_in_check_plane_starting_position() {
        let state = GameState::new();
        let tensor = encode_board(&state);

        // Starting position: not in check, channel 18 should be all 0.
        let ch = 18;
        for row in 0..BOARD_DIM {
            for col in 0..BOARD_DIM {
                assert_eq!(
                    tensor[ch * BOARD_DIM * BOARD_DIM + row * BOARD_DIM + col],
                    0.0,
                    "in-check plane should be 0.0 for starting position"
                );
            }
        }
    }

    #[test]
    fn test_in_check_plane_when_checked() {
        // Set up a position where white king is in check.
        // White king at (0, -4), black rook on same file at (0, 0).
        let mut board = Board::empty();
        board.set(
            HexCoord::new(0, -4),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        board.set(
            HexCoord::new(0, 4),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        board.set(
            HexCoord::new(0, 0),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        board.white_king = HexCoord::new(0, -4);
        board.black_king = HexCoord::new(0, 4);
        board.side_to_move = Color::White;

        let state = GameState::from_board(board);
        let tensor = encode_board(&state);

        // Channel 18 should be a constant 1.0 plane (white is in check).
        let ch = 18;
        for row in 0..BOARD_DIM {
            for col in 0..BOARD_DIM {
                assert_eq!(
                    tensor[ch * BOARD_DIM * BOARD_DIM + row * BOARD_DIM + col],
                    1.0,
                    "in-check plane should be 1.0 when in check"
                );
            }
        }
    }

    #[test]
    fn test_num_move_indices_reasonable() {
        let n = num_move_indices();
        eprintln!("NUM_MOVE_INDICES = {}", n);
        assert!(
            n >= 3000 && n <= 5000,
            "NUM_MOVE_INDICES = {} is outside the expected range [3000, 5000]",
            n
        );
    }

    #[test]
    fn test_move_to_index_roundtrip() {
        let from = HexCoord::new(0, 0);
        let to = HexCoord::new(3, -2); // knight jump
        let mv = Move::new(from, to, None);

        let idx = move_to_index(&mv).expect("move should be in table");
        let (f2, t2, p2) = index_to_move(idx);
        assert_eq!(f2, from);
        assert_eq!(t2, to);
        assert_eq!(p2, None);
    }

    #[test]
    fn test_move_to_index_roundtrip_sliding() {
        let from = HexCoord::new(-5, 0);
        let to = HexCoord::new(-5, 5);
        let mv = Move::new(from, to, None);
        let idx = move_to_index(&mv).expect("move should be in table");
        let (f2, t2, p2) = index_to_move(idx);
        assert_eq!(f2, from);
        assert_eq!(t2, to);
        assert_eq!(p2, None);
    }

    #[test]
    fn test_promotion_moves_get_distinct_indices() {
        let from = HexCoord::new(0, 4);
        let to = HexCoord::new(0, 5);
        assert!(is_promotion_cell(to), "target should be a promotion cell");

        let idx_q =
            move_to_index(&Move::new(from, to, None).with_promotion(PieceKind::Queen)).unwrap();
        let idx_r =
            move_to_index(&Move::new(from, to, None).with_promotion(PieceKind::Rook)).unwrap();
        let idx_b =
            move_to_index(&Move::new(from, to, None).with_promotion(PieceKind::Bishop)).unwrap();
        let idx_n =
            move_to_index(&Move::new(from, to, None).with_promotion(PieceKind::Knight)).unwrap();

        let mut indices = vec![idx_q, idx_r, idx_b, idx_n];
        indices.sort();
        indices.dedup();
        assert_eq!(
            indices.len(),
            4,
            "each promotion piece should get a distinct index"
        );
    }

    /// Golden hash guarding the move-index bijection against accidental changes.
    ///
    /// The move table must stay byte-identical across commits — any change
    /// invalidates every existing training sample and every trained model
    /// (policy head outputs are indexed by this table). If this test fails,
    /// you almost certainly did not mean to change the move encoding. If you
    /// really do need to change it, bump a run_id / model version and update
    /// this constant deliberately.
    #[test]
    fn test_move_table_hash_is_stable() {
        let n = num_move_indices();
        // Simple deterministic FNV-1a-style 64-bit rolling hash over the
        // full (from_idx, to_idx, promotion_ordinal) sequence.
        let mut h: u64 = 0xcbf29ce484222325;
        for i in 0..n {
            let e = &MOVE_INDEX.entries[i];
            for byte in [e.from_idx, e.to_idx, promotion_ordinal(e.promotion)] {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        const EXPECTED_COUNT: usize = 4206;
        const EXPECTED_HASH: u64 = 0x84d424aff5f7c2fd;
        // Print on failure so an intentional update is a one-line change.
        assert_eq!(
            (n, h),
            (EXPECTED_COUNT, EXPECTED_HASH),
            "move table drift detected: got count={}, hash=0x{:016x} — \
             if this change is intentional, update EXPECTED_COUNT/EXPECTED_HASH \
             and bump the training run_id",
            n,
            h
        );
    }

    #[test]
    fn test_all_indices_are_unique() {
        let n = num_move_indices();
        for i in 0..n {
            let (from, to, promo) = index_to_move(i);
            let mv = Move::new(from, to, None).with_promotion_opt(promo);
            let j = move_to_index(&mv).expect("move should be in table");
            assert_eq!(i, j, "index round-trip failed for index {}", i);
        }
    }

    #[test]
    fn test_training_record_serialization_roundtrip() {
        let state = GameState::new();
        let tensor = encode_board(&state);
        let n = num_move_indices();

        let mut policy = vec![0.0f32; n];
        policy[0] = 0.5;
        policy[n - 1] = 0.3;
        if n > 100 {
            policy[100] = 0.2;
        }

        let record = TrainingRecord {
            board_tensor: tensor,
            policy_target: policy.clone(),
            wdl_target: [0.8, 0.15, 0.05],
        };

        let bytes = record.to_bytes();
        assert_eq!(bytes.len(), TrainingRecord::record_size());

        let restored = TrainingRecord::from_bytes(&bytes);
        assert_eq!(restored.board_tensor, tensor);
        assert_eq!(restored.policy_target, policy);
        assert!((restored.wdl_target[0] - 0.8).abs() < 1e-7);
        assert!((restored.wdl_target[1] - 0.15).abs() < 1e-7);
        assert!((restored.wdl_target[2] - 0.05).abs() < 1e-7);
    }

    #[test]
    fn test_training_record_size() {
        let n = num_move_indices();
        let expected = (TENSOR_SIZE + n + 3) * 4;
        assert_eq!(TrainingRecord::record_size(), expected);
    }

    #[test]
    fn test_encode_board_known_piece_placement() {
        // Place a single white rook at (0, 0) and verify exactly that cell is
        // set in channel 3 (White Rook).
        let mut board = Board::empty();
        let coord = HexCoord::new(0, 0);
        board.set(coord, Some(Piece::new(PieceKind::Rook, Color::White)));
        board.white_king = HexCoord::new(-5, 0); // dummy king off to the side
        board.black_king = HexCoord::new(5, 0);

        let state = GameState::from_board(board);
        let tensor = encode_board(&state);
        let ch = 3; // White Rook
        let col = 5;
        let row = 5;
        assert_eq!(
            tensor[ch * BOARD_DIM * BOARD_DIM + row * BOARD_DIM + col],
            1.0
        );

        let mut count = 0;
        for r in 0..BOARD_DIM {
            for c in 0..BOARD_DIM {
                if tensor[ch * BOARD_DIM * BOARD_DIM + r * BOARD_DIM + c] > 0.5 {
                    count += 1;
                }
            }
        }
        assert_eq!(count, 1);
    }
}
