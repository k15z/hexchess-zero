use crate::board::{index_to_coord, Board, Color, HexCoord, PieceKind};

/// Return the material value of a piece kind in centipawns.
pub fn piece_value(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::Pawn => 100,
        PieceKind::Knight => 300,
        PieceKind::Bishop => 300,
        PieceKind::Rook => 500,
        PieceKind::Queen => 900,
        PieceKind::King => 0,
    }
}

/// Count total material for a color in centipawns.
pub fn material(board: &Board, color: Color) -> i32 {
    board.all_pieces(color).map(|(_, p)| piece_value(p.kind)).sum()
}

/// Hex distance from the center (0,0). Returns 0 at center, 5 at edge.
fn hex_distance_from_center(coord: HexCoord) -> i32 {
    let q = coord.q as i32;
    let r = coord.r as i32;
    q.abs().max(r.abs()).max((q + r).abs())
}

/// Evaluate the board position from the perspective of the side to move.
/// Returns centipawns: positive = side to move is better, negative = opponent is better.
pub fn evaluate(board: &Board) -> i32 {
    let us = board.side_to_move;
    let mut mat = 0i32;
    let mut center = 0i32;

    // Single pass over all cells
    for (idx, cell) in board.cells.iter().enumerate() {
        if let Some(piece) = cell {
            let coord = index_to_coord(idx);
            let sign = if piece.color == us { 1 } else { -1 };
            mat += sign * piece_value(piece.kind);
            center += sign * (5 - hex_distance_from_center(coord));
        }
    }

    // King safety
    let us_king_dist = hex_distance_from_center(board.king_pos(us));
    let them_king_dist = hex_distance_from_center(board.king_pos(us.opponent()));
    let king_penalty = |dist: i32| -> i32 {
        if dist >= 5 { -30 } else if dist >= 4 { -15 } else { 0 }
    };
    let safety = king_penalty(us_king_dist) - king_penalty(them_king_dist);

    mat + center + safety
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Color, HexCoord, PieceKind};

    #[test]
    fn starting_position_is_zero() {
        let board = Board::new();
        assert_eq!(evaluate(&board), 0);
    }

    #[test]
    fn removing_white_pawn_hurts_white() {
        let mut board = Board::new();
        board.set(HexCoord::new(0, -1), None);
        let score = evaluate(&board);
        assert!(score < 0, "Expected negative eval after removing white pawn, got {}", score);
    }

    #[test]
    fn material_counting() {
        let board = Board::new();
        let white_mat = material(&board, Color::White);
        let black_mat = material(&board, Color::Black);
        assert_eq!(white_mat, 4300);
        assert_eq!(black_mat, 4300);
    }

    #[test]
    fn piece_values_correct() {
        assert_eq!(piece_value(PieceKind::Pawn), 100);
        assert_eq!(piece_value(PieceKind::Knight), 300);
        assert_eq!(piece_value(PieceKind::Bishop), 300);
        assert_eq!(piece_value(PieceKind::Rook), 500);
        assert_eq!(piece_value(PieceKind::Queen), 900);
        assert_eq!(piece_value(PieceKind::King), 0);
    }

    #[test]
    fn center_distance() {
        assert_eq!(hex_distance_from_center(HexCoord::new(0, 0)), 0);
        assert_eq!(hex_distance_from_center(HexCoord::new(1, 0)), 1);
        assert_eq!(hex_distance_from_center(HexCoord::new(0, 5)), 5);
        assert_eq!(hex_distance_from_center(HexCoord::new(-5, 0)), 5);
        assert_eq!(hex_distance_from_center(HexCoord::new(3, -3)), 3);
    }
}
