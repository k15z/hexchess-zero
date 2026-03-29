use crate::board::{Board, Color, PieceKind};
use crate::game::{GameState, GameStatus};

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

/// Evaluate a game state from the perspective of the side to move.
/// Terminal states return large values (±10000 for checkmate, 0 for draws).
/// Non-terminal states return material difference in centipawns.
pub fn evaluate(state: &GameState) -> i32 {
    match state.status() {
        GameStatus::Checkmate(winner) => {
            if winner == state.side_to_move() {
                10_000
            } else {
                -10_000
            }
        }
        GameStatus::Stalemate
        | GameStatus::DrawByRepetition
        | GameStatus::DrawByFiftyMoves
        | GameStatus::DrawByInsufficientMaterial => 0,
        GameStatus::Ongoing => {
            let us = state.side_to_move();
            let mut mat = 0i32;
            for cell in state.board.cells.iter().flatten() {
                let sign = if cell.color == us { 1 } else { -1 };
                mat += sign * piece_value(cell.kind);
            }
            mat
        }
    }
}

/// Evaluate just the board material (no terminal check). Used when you
/// already know the game is ongoing.
pub fn evaluate_board(board: &Board) -> i32 {
    let us = board.side_to_move;
    board.cells.iter().flatten().map(|p| {
        let sign = if p.color == us { 1 } else { -1 };
        sign * piece_value(p.kind)
    }).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Color, HexCoord, PieceKind};

    #[test]
    fn starting_position_is_zero() {
        let state = GameState::new();
        assert_eq!(evaluate(&state), 0);
    }

    #[test]
    fn removing_white_pawn_hurts_white() {
        let mut state = GameState::new();
        state.board.set(HexCoord::new(0, -1), None);
        let score = evaluate(&state);
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
}
