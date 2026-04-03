use crate::board::{BOARD_RADIUS, Board, CARDINAL_DIRS, Color, HexCoord, PieceKind};
use crate::game::{GameState, GameStatus};
use crate::movegen;

/// Weights for each evaluation signal. Set a weight to 0 to disable that signal.
#[derive(Clone, Copy, Debug)]
pub struct EvalWeights {
    pub material: i32,
    pub mobility: i32,
    pub pawn_advance: i32,
    pub center_control: i32,
    pub king_safety: i32,
    pub bishop_color_bonus: i32,
    pub pawn_connected: i32,
    pub pawn_isolated: i32,
    pub passed_pawn: i32,
    pub king_tropism: i32,
}

impl Default for EvalWeights {
    fn default() -> Self {
        Self {
            material: 1,
            mobility: 0,
            pawn_advance: 1,
            center_control: 4,
            king_safety: 0,
            bishop_color_bonus: 30,
            pawn_connected: 7,
            pawn_isolated: -10,
            passed_pawn: 3,
            king_tropism: 2,
        }
    }
}

impl EvalWeights {
    pub fn material_only() -> Self {
        Self {
            material: 1,
            ..ZEROED
        }
    }
}

const ZEROED: EvalWeights = EvalWeights {
    material: 0,
    mobility: 0,
    pawn_advance: 0,
    center_control: 0,
    king_safety: 0,
    bishop_color_bonus: 0,
    pawn_connected: 0,
    pawn_isolated: 0,
    passed_pawn: 0,
    king_tropism: 0,
};

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
    board
        .all_pieces(color)
        .map(|(_, p)| piece_value(p.kind))
        .sum()
}

// ---------------------------------------------------------------------------
// Signal helpers
// ---------------------------------------------------------------------------

/// Score a signal for both sides: f(us) - f(opponent).
fn differential<F: Fn(Color) -> i32>(us: Color, f: F) -> i32 {
    f(us) - f(us.opponent())
}

fn hex_dist(coord: HexCoord) -> i32 {
    let q = coord.q as i32;
    let r = coord.r as i32;
    let s = -q - r;
    q.abs().max(r.abs()).max(s.abs())
}

fn hex_dist_between(a: HexCoord, b: HexCoord) -> i32 {
    let dq = (a.q as i32 - b.q as i32).abs();
    let dr = (a.r as i32 - b.r as i32).abs();
    let ds = ((a.q as i32 + a.r as i32) - (b.q as i32 + b.r as i32)).abs();
    dq.max(dr).max(ds)
}

const ADVANCE_BONUS: [i32; 11] = [0, 64, 32, 16, 8, 4, 2, 1, 1, 1, 1];

fn distance_to_promotion(q: i8, r: i8, color: Color) -> usize {
    let dist = match color {
        Color::White => {
            let promo_r = if q < 0 { 5 } else { 5 - q as i32 };
            promo_r - r as i32
        }
        Color::Black => {
            let promo_r = if q > 0 { -5 } else { -5 - q as i32 };
            r as i32 - promo_r
        }
    };
    dist.max(0) as usize
}

fn advance_bonus(q: i8, r: i8, color: Color) -> i32 {
    let dist = distance_to_promotion(q, r, color);
    ADVANCE_BONUS[dist.min(ADVANCE_BONUS.len() - 1)]
}

// ---------------------------------------------------------------------------
// Individual signal functions
// ---------------------------------------------------------------------------

fn mobility_score(board: &Board, us: Color) -> i32 {
    // One side matches board.side_to_move; the other needs a clone-and-flip.
    let direct_count = movegen::generate_pseudo_legal_moves(board).len() as i32;
    let mut flipped = board.clone();
    flipped.side_to_move = us.opponent();
    let flipped_count = movegen::generate_pseudo_legal_moves(&flipped).len() as i32;

    if board.side_to_move == us {
        direct_count - flipped_count
    } else {
        flipped_count - direct_count
    }
}

fn pawn_advance_score(board: &Board, us: Color) -> i32 {
    differential(us, |color| {
        board
            .all_pieces(color)
            .filter(|(_, p)| p.kind == PieceKind::Pawn)
            .map(|(c, _)| advance_bonus(c.q, c.r, color))
            .sum()
    })
}

fn passed_pawn_score(board: &Board, us: Color) -> i32 {
    // Pre-collect pawn positions for both sides to avoid repeated 91-cell scans.
    let mut white_pawns: [(i8, i8); 9] = [(0, 0); 9];
    let mut black_pawns: [(i8, i8); 9] = [(0, 0); 9];
    let mut n_white = 0usize;
    let mut n_black = 0usize;
    for (c, p) in board.all_pieces(Color::White) {
        if p.kind == PieceKind::Pawn && n_white < 9 {
            white_pawns[n_white] = (c.q, c.r);
            n_white += 1;
        }
    }
    for (c, p) in board.all_pieces(Color::Black) {
        if p.kind == PieceKind::Pawn && n_black < 9 {
            black_pawns[n_black] = (c.q, c.r);
            n_black += 1;
        }
    }

    let is_passed = |q: i8, r: i8, color: Color| -> bool {
        let (opp_pawns, n_opp) = match color {
            Color::White => (&black_pawns, n_black),
            Color::Black => (&white_pawns, n_white),
        };
        !opp_pawns[..n_opp].iter().any(|&(cq, cr)| {
            (cq - q).abs() <= 1
                && match color {
                    Color::White => cr > r,
                    Color::Black => cr < r,
                }
        })
    };

    let score_for = |pawns: &[(i8, i8)], n: usize, color: Color| -> i32 {
        pawns[..n]
            .iter()
            .filter(|&&(q, r)| is_passed(q, r, color))
            .map(|&(q, r)| advance_bonus(q, r, color))
            .sum()
    };

    let our = match us {
        Color::White => score_for(&white_pawns, n_white, Color::White),
        Color::Black => score_for(&black_pawns, n_black, Color::Black),
    };
    let their = match us.opponent() {
        Color::White => score_for(&white_pawns, n_white, Color::White),
        Color::Black => score_for(&black_pawns, n_black, Color::Black),
    };
    our - their
}

fn king_tropism_score(board: &Board, us: Color) -> i32 {
    let enemy_king = board.king_pos(us.opponent());
    let our_king = board.king_pos(us);

    let tropism_for = |color: Color, target: HexCoord| -> i32 {
        board
            .all_pieces(color)
            .filter(|(_, p)| {
                matches!(
                    p.kind,
                    PieceKind::Knight | PieceKind::Bishop | PieceKind::Rook | PieceKind::Queen
                )
            })
            .map(|(c, _)| (7 - hex_dist_between(c, target)).max(0))
            .sum()
    };

    tropism_for(us, enemy_king) - tropism_for(us.opponent(), our_king)
}

fn center_control_score(board: &Board, us: Color) -> i32 {
    let centrality_weight = |kind: PieceKind| -> i32 {
        match kind {
            PieceKind::Knight => 2,
            PieceKind::Pawn | PieceKind::Bishop | PieceKind::Queen => 1,
            PieceKind::Rook | PieceKind::King => 0,
        }
    };
    let radius = BOARD_RADIUS as i32;

    let score_for = |color: Color| -> i32 {
        board
            .all_pieces(color)
            .map(|(coord, piece)| {
                let w = centrality_weight(piece.kind);
                w * (radius - hex_dist(coord))
            })
            .sum()
    };

    score_for(us) - score_for(us.opponent())
}

fn king_safety_score(board: &Board, us: Color) -> i32 {
    let exposure = |color: Color| -> i32 {
        let king = board.king_pos(color);
        let attacker = color.opponent();
        CARDINAL_DIRS
            .iter()
            .filter_map(|&(dq, dr)| king.step(dq, dr))
            .filter(|&n| movegen::is_square_attacked(board, n, attacker))
            .count() as i32
    };

    exposure(us.opponent()) - exposure(us)
}

fn hex_cell_color(coord: HexCoord) -> usize {
    ((coord.q as i32 - coord.r as i32) % 3 + 3) as usize % 3
}

fn bishop_color_score(board: &Board, us: Color) -> i32 {
    let count_colors = |color: Color| -> i32 {
        let mut seen = [false; 3];
        for (coord, piece) in board.all_pieces(color) {
            if piece.kind == PieceKind::Bishop {
                seen[hex_cell_color(coord)] = true;
            }
        }
        seen.iter().filter(|&&s| s).count() as i32
    };

    let our_colors = count_colors(us);
    let their_colors = count_colors(us.opponent());
    (our_colors - 1).max(0) - (their_colors - 1).max(0)
}

fn pawn_structure_score(board: &Board, us: Color) -> (i32, i32) {
    fn score_for_color(board: &Board, color: Color) -> (i32, i32) {
        let defend_offsets: [(i8, i8); 2] = match color {
            Color::White => [(1, -1), (-1, 0)],
            Color::Black => [(-1, 1), (1, 0)],
        };

        // Precompute which q-columns have pawns for O(1) isolated-pawn check.
        let mut pawn_on_file = [false; 11]; // q ranges -5..5, index = q+5
        for (coord, piece) in board.all_pieces(color) {
            if piece.kind == PieceKind::Pawn {
                pawn_on_file[(coord.q + 5) as usize] = true;
            }
        }

        let mut connected = 0i32;
        let mut isolated = 0i32;

        for (coord, piece) in board.all_pieces(color) {
            if piece.kind != PieceKind::Pawn {
                continue;
            }

            let is_connected = defend_offsets.iter().any(|&(dq, dr)| {
                let check = HexCoord::new(coord.q + dq, coord.r + dr);
                check.is_valid()
                    && matches!(board.get(check), Some(p) if p.color == color && p.kind == PieceKind::Pawn)
            });
            if is_connected {
                connected += 1;
            }

            let qi = (coord.q + 5) as usize;
            let has_neighbor =
                (qi > 0 && pawn_on_file[qi - 1]) || (qi < 10 && pawn_on_file[qi + 1]);
            if !has_neighbor {
                isolated += 1;
            }
        }

        (connected, isolated)
    }

    let (our_conn, our_iso) = score_for_color(board, us);
    let (their_conn, their_iso) = score_for_color(board, us.opponent());
    (our_conn - their_conn, our_iso - their_iso)
}

// ---------------------------------------------------------------------------
// Weighted evaluation
// ---------------------------------------------------------------------------

/// Evaluate a board position with configurable signal weights.
/// Returns score in centipawns from the side-to-move's perspective.
pub fn evaluate_board_weighted(board: &Board, weights: &EvalWeights) -> i32 {
    let us = board.side_to_move;
    let mut score = 0i32;

    if weights.material != 0 {
        score += weights.material * evaluate_board(board);
    }
    if weights.mobility != 0 {
        score += weights.mobility * mobility_score(board, us);
    }
    if weights.pawn_advance != 0 {
        score += weights.pawn_advance * pawn_advance_score(board, us);
    }
    if weights.center_control != 0 {
        score += weights.center_control * center_control_score(board, us);
    }
    if weights.king_safety != 0 {
        score += weights.king_safety * king_safety_score(board, us);
    }
    if weights.bishop_color_bonus != 0 {
        score += weights.bishop_color_bonus * bishop_color_score(board, us);
    }
    if weights.pawn_connected != 0 || weights.pawn_isolated != 0 {
        let (conn, iso) = pawn_structure_score(board, us);
        score += weights.pawn_connected * conn;
        score += weights.pawn_isolated * iso;
    }
    if weights.passed_pawn != 0 {
        score += weights.passed_pawn * passed_pawn_score(board, us);
    }
    if weights.king_tropism != 0 {
        score += weights.king_tropism * king_tropism_score(board, us);
    }

    score
}

// ---------------------------------------------------------------------------
// Original evaluation functions (backward compatibility)
// ---------------------------------------------------------------------------

/// Evaluate a game state from the perspective of the side to move.
/// Terminal states return large values (±10000 for checkmate, 0 for draws).
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
        GameStatus::Ongoing => evaluate_board(&state.board),
    }
}

/// Evaluate just the board material (no terminal check).
pub fn evaluate_board(board: &Board) -> i32 {
    let us = board.side_to_move;
    board
        .cells
        .iter()
        .flatten()
        .map(|p| {
            let sign = if p.color == us { 1 } else { -1 };
            sign * piece_value(p.kind)
        })
        .sum()
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
        assert!(
            score < 0,
            "Expected negative eval after removing white pawn, got {}",
            score
        );
    }

    #[test]
    fn material_counting() {
        let board = Board::new();
        assert_eq!(material(&board, Color::White), 4300);
        assert_eq!(material(&board, Color::Black), 4300);
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
    fn weighted_eval_starting_position_is_zero() {
        let board = Board::new();
        assert_eq!(evaluate_board_weighted(&board, &EvalWeights::default()), 0);
    }

    #[test]
    fn weighted_eval_material_only_matches_original() {
        let board = Board::new();
        let weights = EvalWeights::material_only();
        assert_eq!(
            evaluate_board_weighted(&board, &weights),
            evaluate_board(&board)
        );
    }

    #[test]
    fn pawn_advance_increases_after_push() {
        let mut state = GameState::new();
        let before = pawn_advance_score(&state.board, Color::White);
        let moves = state.legal_moves();
        let pawn_move = moves
            .iter()
            .find(|m| matches!(state.board.get(m.from), Some(p) if p.kind == PieceKind::Pawn));
        if let Some(mv) = pawn_move {
            state.apply_move(mv.clone());
            let after = pawn_advance_score(&state.board, Color::White);
            assert!(after > before, "before={}, after={}", before, after);
        }
    }

    #[test]
    fn passed_pawn_symmetric_at_start() {
        let board = Board::new();
        assert_eq!(passed_pawn_score(&board, Color::White), 0);
    }

    #[test]
    fn king_tropism_symmetric_at_start() {
        let board = Board::new();
        assert_eq!(king_tropism_score(&board, Color::White), 0);
    }

    #[test]
    fn distance_to_promotion_correct() {
        assert_eq!(distance_to_promotion(0, -1, Color::White), 6);
        assert_eq!(distance_to_promotion(0, 4, Color::White), 1);
        assert_eq!(distance_to_promotion(0, 1, Color::Black), 6);
    }

    #[test]
    fn bishop_color_starting_position() {
        let board = Board::new();
        assert_eq!(bishop_color_score(&board, Color::White), 0);
    }

    #[test]
    fn center_control_symmetric_at_start() {
        let board = Board::new();
        assert_eq!(center_control_score(&board, Color::White), 0);
    }

    #[test]
    fn king_safety_symmetric_at_start() {
        let board = Board::new();
        assert_eq!(king_safety_score(&board, Color::White), 0);
    }
}
