use crate::board::{Board, Color, HexCoord, Piece, PieceKind, ZOBRIST, coord_to_index};
use crate::movegen::{self, Move};

// ---------------------------------------------------------------------------
// GameStatus
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameStatus {
    Ongoing,
    Checkmate(Color), // Color is the **winner**
    Stalemate,
    DrawByRepetition,
    DrawByFiftyMoves,
    DrawByInsufficientMaterial,
}

impl GameStatus {
    /// Lowercase string for FFI/serialization.
    pub fn as_str(self) -> &'static str {
        match self {
            GameStatus::Ongoing => "ongoing",
            GameStatus::Checkmate(Color::White) => "checkmate_white",
            GameStatus::Checkmate(Color::Black) => "checkmate_black",
            GameStatus::Stalemate => "stalemate",
            GameStatus::DrawByRepetition => "draw_repetition",
            GameStatus::DrawByFiftyMoves => "draw_fifty",
            GameStatus::DrawByInsufficientMaterial => "draw_material",
        }
    }
}

// ---------------------------------------------------------------------------
// UndoInfo — everything we need to restore the position after undo
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct UndoInfo {
    mv: Move,
    en_passant: Option<HexCoord>, // en-passant state *before* this move
    halfmove_clock: u16,          // halfmove clock *before* this move
    zobrist_hash: u64,            // zobrist hash *before* this move
}

// ---------------------------------------------------------------------------
// GameState
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct GameState {
    pub board: Board,
    move_history: Vec<UndoInfo>,
    /// Zobrist hashes after every half-move, used for repetition detection.
    position_history: Vec<u64>,
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create a new game from the standard Glinski starting position.
    pub fn new() -> Self {
        let board = Board::new();
        let hash = board.zobrist_hash;
        Self {
            board,
            move_history: Vec::new(),
            position_history: vec![hash],
        }
    }

    /// Create a game from a given board (useful for testing).
    pub fn from_board(board: Board) -> Self {
        let hash = board.zobrist_hash;
        Self {
            board,
            move_history: Vec::new(),
            position_history: vec![hash],
        }
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Side to move.
    pub fn side_to_move(&self) -> Color {
        self.board.side_to_move
    }

    /// Number of half-moves (plies) played so far.
    pub fn move_count(&self) -> usize {
        self.move_history.len()
    }

    /// All legal moves in the current position.
    pub fn legal_moves(&self) -> Vec<Move> {
        movegen::generate_legal_moves(&self.board).to_vec()
    }

    /// Is the game over (any terminal status)?
    pub fn is_game_over(&self) -> bool {
        self.status() != GameStatus::Ongoing
    }

    /// Outcome from the **side to move's** perspective.
    /// +1.0 = side to move won, -1.0 = side to move lost, 0.0 = draw, None = ongoing.
    pub fn outcome_value(&self) -> Option<f32> {
        match self.status() {
            GameStatus::Ongoing => None,
            GameStatus::Checkmate(winner) => {
                if winner == self.side_to_move() {
                    Some(1.0) // side to move won
                } else {
                    Some(-1.0) // side to move lost (they're checkmated)
                }
            }
            _ => Some(0.0),
        }
    }

    // ------------------------------------------------------------------
    // Status determination
    // ------------------------------------------------------------------

    pub fn status(&self) -> GameStatus {
        // 1. Fifty-move rule (100 half-moves)
        if self.board.halfmove_clock >= 100 {
            return GameStatus::DrawByFiftyMoves;
        }

        // 2. Three-fold repetition
        if self.is_draw_by_repetition() {
            return GameStatus::DrawByRepetition;
        }

        // 3. Insufficient material
        if self.is_insufficient_material() {
            return GameStatus::DrawByInsufficientMaterial;
        }

        // 4. Generate legal moves for the side to move
        let moves = self.legal_moves();
        if moves.is_empty() {
            let stm = self.side_to_move();
            if movegen::is_in_check(&self.board, stm) {
                // The opponent delivered checkmate
                return GameStatus::Checkmate(stm.opponent());
            } else {
                return GameStatus::Stalemate;
            }
        }

        GameStatus::Ongoing
    }

    /// Check whether the current position has appeared at least 3 times.
    fn is_draw_by_repetition(&self) -> bool {
        let current = self.board.zobrist_hash;
        let count = self
            .position_history
            .iter()
            .filter(|&&h| h == current)
            .count();
        count >= 3
    }

    /// Basic insufficient-material check: K vs K, K+B vs K.
    fn is_insufficient_material(&self) -> bool {
        let mut non_king_count = 0u8;
        let mut lone_kind = PieceKind::Pawn; // placeholder

        for cell in &self.board.cells {
            if let Some(piece) = cell
                && piece.kind != PieceKind::King
            {
                non_king_count += 1;
                if non_king_count > 1 {
                    return false;
                }
                lone_kind = piece.kind;
            }
        }

        // K vs K, or K+B vs K
        non_king_count == 0 || (non_king_count == 1 && lone_kind == PieceKind::Bishop)
    }

    // ------------------------------------------------------------------
    // Apply / Undo moves
    // ------------------------------------------------------------------

    pub fn apply_move(&mut self, mv: Move) {
        // 1. Save undo info (state *before* this move)
        let undo = UndoInfo {
            mv,
            en_passant: self.board.en_passant,
            halfmove_clock: self.board.halfmove_clock,
            zobrist_hash: self.board.zobrist_hash,
        };

        let side = self.board.side_to_move;

        // 2. Pick up the piece at `from`
        let mut piece = self
            .board
            .get(mv.from)
            .expect("apply_move: no piece at `from`");
        self.board.set(mv.from, None);

        // 3. Handle en passant capture
        if mv.is_en_passant {
            // The captured pawn is the one that double-moved past the ep target.
            // White captures a black pawn (which moved down, so it's below the target).
            // Black captures a white pawn (which moved up, so it's above the target).
            let captured_pawn_coord = match side {
                Color::White => HexCoord::new(mv.to.q, mv.to.r - 1),
                Color::Black => HexCoord::new(mv.to.q, mv.to.r + 1),
            };
            self.board.set(captured_pawn_coord, None);
        }

        // 4. Handle promotion
        if let Some(promo_kind) = mv.promotion {
            piece = Piece {
                kind: promo_kind,
                color: side,
            };
        }

        // 5. Place piece on destination
        self.board.set(mv.to, Some(piece));

        // 6. Update en passant target
        self.board.en_passant = None;
        if piece.kind == PieceKind::Pawn {
            let dq = mv.to.q - mv.from.q;
            let dr = mv.to.r - mv.from.r;
            // Detect a double pawn push (straight advance of 2 along r-axis)
            if dq == 0 && (dr == 2 || dr == -2) {
                let intermediate = HexCoord::new(mv.from.q, mv.from.r + dr / 2);
                self.board.en_passant = Some(intermediate);
            }
        }

        // 7. Update Zobrist hash for en passant change
        if let Some(old_ep) = undo.en_passant
            && let Some(idx) = coord_to_index(old_ep)
        {
            self.board.zobrist_hash ^= ZOBRIST.en_passant[idx];
        }
        if let Some(new_ep) = self.board.en_passant
            && let Some(idx) = coord_to_index(new_ep)
        {
            self.board.zobrist_hash ^= ZOBRIST.en_passant[idx];
        }

        // 8. Update clocks
        if piece.kind == PieceKind::Pawn || mv.captured.is_some() || mv.is_en_passant {
            self.board.halfmove_clock = 0;
        } else {
            self.board.halfmove_clock += 1;
        }
        if side == Color::Black {
            self.board.fullmove_number += 1;
        }

        // 9. Switch side to move and update Zobrist
        self.board.side_to_move = side.opponent();
        self.board.zobrist_hash ^= ZOBRIST.side_to_move;

        // 10. Record position hash for repetition detection
        self.position_history.push(self.board.zobrist_hash);

        // 11. Push undo info
        self.move_history.push(undo);
    }

    pub fn undo_move(&mut self) {
        let undo = self.move_history.pop().expect("undo_move: no move to undo");
        let mv = &undo.mv;

        // Pop position history (the hash pushed by apply_move)
        self.position_history.pop();

        // Switch side back
        self.board.side_to_move = self.board.side_to_move.opponent();
        let side = self.board.side_to_move; // side that made the move we're undoing

        // Determine what piece is currently on `to`
        let mut piece = self.board.get(mv.to).expect("undo_move: no piece at `to`");

        // Undo promotion: restore the piece to a pawn
        if mv.promotion.is_some() {
            piece = Piece {
                kind: PieceKind::Pawn,
                color: side,
            };
        }

        // Move piece back to `from`
        self.board.set(mv.from, Some(piece));

        // Restore the captured piece (or clear `to`)
        if mv.is_en_passant {
            // `to` was empty (the ep target); captured pawn was on an adjacent square
            self.board.set(mv.to, None);
            let captured_pawn_coord = match side {
                Color::White => HexCoord::new(mv.to.q, mv.to.r - 1),
                Color::Black => HexCoord::new(mv.to.q, mv.to.r + 1),
            };
            // EP always captures an opponent pawn
            let captured_pawn = Some(Piece::new(PieceKind::Pawn, side.opponent()));
            self.board.set(captured_pawn_coord, captured_pawn);
        } else {
            // Normal capture or quiet move: put captured piece (or None) back on `to`
            self.board.set(mv.to, mv.captured);
        }

        // Restore clocks / state saved in UndoInfo
        self.board.en_passant = undo.en_passant;
        self.board.halfmove_clock = undo.halfmove_clock;
        self.board.zobrist_hash = undo.zobrist_hash;

        // Undo fullmove increment (was incremented after Black's move)
        if side == Color::Black {
            self.board.fullmove_number -= 1;
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- helpers -----

    /// Build a minimal board with only kings.
    fn kings_only_board() -> Board {
        let mut board = Board::empty();
        board.set(
            HexCoord::new(0, -4),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        board.white_king = HexCoord::new(0, -4);
        board.set(
            HexCoord::new(0, 4),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        board.black_king = HexCoord::new(0, 4);
        board
    }

    fn make_quiet_move(from: (i8, i8), to: (i8, i8)) -> Move {
        Move {
            from: HexCoord::new(from.0, from.1),
            to: HexCoord::new(to.0, to.1),
            promotion: None,
            captured: None,
            is_en_passant: false,
        }
    }

    fn make_capture_move(from: (i8, i8), to: (i8, i8), captured: Piece) -> Move {
        Move {
            from: HexCoord::new(from.0, from.1),
            to: HexCoord::new(to.0, to.1),
            promotion: None,
            captured: Some(captured),
            is_en_passant: false,
        }
    }

    // ----- test: initial status is Ongoing -----

    #[test]
    fn test_initial_status_is_ongoing() {
        let game = GameState::new();
        assert_eq!(game.status(), GameStatus::Ongoing);
    }

    // ----- test: move count tracking -----

    #[test]
    fn test_move_count() {
        let mut board = kings_only_board();
        board.set(
            HexCoord::new(2, -5),
            Some(Piece::new(PieceKind::Rook, Color::White)),
        );
        board.set(
            HexCoord::new(-2, 5),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        let mut game = GameState::from_board(board);

        assert_eq!(game.move_count(), 0);

        game.apply_move(make_quiet_move((2, -5), (2, -3)));
        assert_eq!(game.move_count(), 1);
        assert_eq!(game.side_to_move(), Color::Black);

        game.apply_move(make_quiet_move((-2, 5), (-2, 3)));
        assert_eq!(game.move_count(), 2);
        assert_eq!(game.side_to_move(), Color::White);
    }

    // ----- test: apply + undo round-trip restores board -----

    #[test]
    fn test_apply_undo_roundtrip_quiet() {
        let mut board = kings_only_board();
        board.set(
            HexCoord::new(2, -5),
            Some(Piece {
                kind: PieceKind::Rook,
                color: Color::White,
            }),
        );
        let mut game = GameState::from_board(board);

        // Snapshot the board cells before the move
        let cells_before = game.board.cells;
        let side_before = game.board.side_to_move;
        let ep_before = game.board.en_passant;
        let hmc_before = game.board.halfmove_clock;
        let fmn_before = game.board.fullmove_number;

        let mv = make_quiet_move((2, -5), (2, -3));
        game.apply_move(mv);

        // After apply, board should differ
        assert_ne!(game.board.side_to_move, side_before);

        game.undo_move();

        // After undo, board should be restored
        assert_eq!(game.board.cells, cells_before);
        assert_eq!(game.board.side_to_move, side_before);
        assert_eq!(game.board.en_passant, ep_before);
        assert_eq!(game.board.halfmove_clock, hmc_before);
        assert_eq!(game.board.fullmove_number, fmn_before);
        assert_eq!(game.move_count(), 0);
    }

    #[test]
    fn test_apply_undo_roundtrip_capture() {
        let mut board = kings_only_board();
        board.set(
            HexCoord::new(2, -3),
            Some(Piece {
                kind: PieceKind::Rook,
                color: Color::White,
            }),
        );
        let black_knight = Piece {
            kind: PieceKind::Knight,
            color: Color::Black,
        };
        board.set(HexCoord::new(2, 0), Some(black_knight));

        let mut game = GameState::from_board(board);
        let cells_before = game.board.cells;

        let mv = make_capture_move((2, -3), (2, 0), black_knight);
        game.apply_move(mv);

        // The black knight should be gone, white rook on (2,0)
        assert_eq!(
            game.board.get(HexCoord::new(2, 0)),
            Some(Piece {
                kind: PieceKind::Rook,
                color: Color::White
            }),
        );
        assert_eq!(game.board.get(HexCoord::new(2, -3)), None);

        game.undo_move();
        assert_eq!(game.board.cells, cells_before);
    }

    // ----- test: en passant apply + undo -----

    #[test]
    fn test_en_passant_apply_undo() {
        let mut board = kings_only_board();
        // White pawn that will double-push
        board.set(
            HexCoord::new(0, -1),
            Some(Piece {
                kind: PieceKind::Pawn,
                color: Color::White,
            }),
        );
        let mut game = GameState::from_board(board);

        // White pawn double-push: (0,-1) -> (0,1)
        let double_push = Move {
            from: HexCoord::new(0, -1),
            to: HexCoord::new(0, 1),
            promotion: None,
            captured: None,
            is_en_passant: false,
        };
        game.apply_move(double_push);

        // En passant target should be the intermediate square
        assert_eq!(game.board.en_passant, Some(HexCoord::new(0, 0)));

        // Now set up a black pawn that could capture ep
        game.board.set(
            HexCoord::new(1, 0),
            Some(Piece {
                kind: PieceKind::Pawn,
                color: Color::Black,
            }),
        );

        let cells_before_ep = game.board.cells;

        let ep_capture = Move {
            from: HexCoord::new(1, 0),
            to: HexCoord::new(0, 0),
            promotion: None,
            captured: Some(Piece {
                kind: PieceKind::Pawn,
                color: Color::White,
            }),
            is_en_passant: true,
        };
        game.apply_move(ep_capture);

        // The white pawn at (0,1) should be removed, black pawn at (0,0)
        assert_eq!(game.board.get(HexCoord::new(0, 1)), None);
        assert_eq!(
            game.board.get(HexCoord::new(0, 0)),
            Some(Piece {
                kind: PieceKind::Pawn,
                color: Color::Black
            }),
        );

        game.undo_move();
        assert_eq!(game.board.cells, cells_before_ep);
    }

    // ----- test: promotion apply + undo -----

    #[test]
    fn test_promotion_apply_undo() {
        let mut board = kings_only_board();
        // White pawn about to promote (needs to be near the far edge)
        board.set(
            HexCoord::new(0, 4),
            None, // remove black king from there first
        );
        board.black_king = HexCoord::new(3, 2); // move black king elsewhere
        board.set(
            HexCoord::new(3, 2),
            Some(Piece {
                kind: PieceKind::King,
                color: Color::Black,
            }),
        );
        board.set(
            HexCoord::new(0, 4),
            Some(Piece {
                kind: PieceKind::Pawn,
                color: Color::White,
            }),
        );
        let mut game = GameState::from_board(board);
        let cells_before = game.board.cells;

        let promo_move = Move {
            from: HexCoord::new(0, 4),
            to: HexCoord::new(0, 5),
            promotion: Some(PieceKind::Queen),
            captured: None,
            is_en_passant: false,
        };
        game.apply_move(promo_move);

        assert_eq!(
            game.board.get(HexCoord::new(0, 5)),
            Some(Piece {
                kind: PieceKind::Queen,
                color: Color::White
            }),
        );

        game.undo_move();
        assert_eq!(game.board.cells, cells_before);
    }

    // ----- test: K vs K is insufficient material -----

    #[test]
    fn test_k_vs_k_is_draw() {
        let board = kings_only_board();
        let game = GameState::from_board(board);
        assert_eq!(game.status(), GameStatus::DrawByInsufficientMaterial);
    }

    // ----- test: draw by repetition -----

    #[test]
    fn test_draw_by_repetition() {
        // Use a real position with rooks shuffling back and forth to create
        // a threefold repetition. Kings at (0,-4) and (0,4), but K vs K + R
        // is NOT insufficient material, so it won't short-circuit.
        let mut board = kings_only_board();
        board.set(
            HexCoord::new(2, -5),
            Some(Piece::new(PieceKind::Rook, Color::White)),
        );
        board.set(
            HexCoord::new(-2, 5),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        let mut game = GameState::from_board(board);

        // Shuttle rooks back and forth to repeat the starting position.
        // Position 1 (initial): Rw(2,-5) Rb(-2,5)
        // Move 1: Rw (2,-5)->(2,-3)  Move 2: Rb (-2,5)->(-2,3)
        // Move 3: Rw (2,-3)->(2,-5)  Move 4: Rb (-2,3)->(-2,5) => Position 1 again (2nd time)
        // Move 5: Rw (2,-5)->(2,-3)  Move 6: Rb (-2,5)->(-2,3)
        // Move 7: Rw (2,-3)->(2,-5)  Move 8: Rb (-2,3)->(-2,5) => Position 1 again (3rd time)

        // Before any repetition
        assert!(!game.is_draw_by_repetition());

        game.apply_move(make_quiet_move((2, -5), (2, -3)));
        game.apply_move(make_quiet_move((-2, 5), (-2, 3)));
        game.apply_move(make_quiet_move((2, -3), (2, -5)));
        game.apply_move(make_quiet_move((-2, 3), (-2, 5)));
        // Second occurrence of starting position
        assert!(!game.is_draw_by_repetition()); // need 3

        game.apply_move(make_quiet_move((2, -5), (2, -3)));
        game.apply_move(make_quiet_move((-2, 5), (-2, 3)));
        game.apply_move(make_quiet_move((2, -3), (2, -5)));
        game.apply_move(make_quiet_move((-2, 3), (-2, 5)));
        // Third occurrence of starting position
        assert!(game.is_draw_by_repetition());
    }

    // ----- test: draw by fifty moves -----

    #[test]
    fn test_draw_by_fifty_moves() {
        let mut board = kings_only_board();
        board.halfmove_clock = 100;
        let game = GameState::from_board(board);
        assert_eq!(game.status(), GameStatus::DrawByFiftyMoves);
    }

    // ----- test: insufficient material K vs K -----

    #[test]
    fn test_insufficient_material_k_vs_k() {
        let board = kings_only_board();
        let game = GameState::from_board(board);
        assert!(game.is_insufficient_material());
    }

    // ----- test: insufficient material K+B vs K -----

    #[test]
    fn test_insufficient_material_kb_vs_k() {
        let mut board = kings_only_board();
        board.set(
            HexCoord::new(1, -3),
            Some(Piece {
                kind: PieceKind::Bishop,
                color: Color::White,
            }),
        );
        let game = GameState::from_board(board);
        assert!(game.is_insufficient_material());
    }

    // ----- test: sufficient material (rook present) -----

    #[test]
    fn test_sufficient_material_with_rook() {
        let mut board = kings_only_board();
        board.set(
            HexCoord::new(1, -3),
            Some(Piece {
                kind: PieceKind::Rook,
                color: Color::White,
            }),
        );
        let game = GameState::from_board(board);
        assert!(!game.is_insufficient_material());
    }

    // ----- test: side_to_move toggles -----

    #[test]
    fn test_side_to_move_toggles() {
        let mut board = kings_only_board();
        board.set(
            HexCoord::new(2, -5),
            Some(Piece {
                kind: PieceKind::Rook,
                color: Color::White,
            }),
        );
        board.set(
            HexCoord::new(-2, 5),
            Some(Piece {
                kind: PieceKind::Rook,
                color: Color::Black,
            }),
        );
        let mut game = GameState::from_board(board);

        assert_eq!(game.side_to_move(), Color::White);
        game.apply_move(make_quiet_move((2, -5), (2, -3)));
        assert_eq!(game.side_to_move(), Color::Black);
        game.apply_move(make_quiet_move((-2, 5), (-2, 3)));
        assert_eq!(game.side_to_move(), Color::White);
    }

    // ----- test: halfmove clock reset on pawn move -----

    #[test]
    fn test_halfmove_clock_reset_on_pawn_move() {
        let mut board = kings_only_board();
        // Place rook and pawn
        board.set(
            HexCoord::new(2, -5),
            Some(Piece {
                kind: PieceKind::Rook,
                color: Color::White,
            }),
        );
        board.set(
            HexCoord::new(0, -1),
            Some(Piece {
                kind: PieceKind::Pawn,
                color: Color::White,
            }),
        );
        board.set(
            HexCoord::new(-2, 5),
            Some(Piece {
                kind: PieceKind::Rook,
                color: Color::Black,
            }),
        );
        let mut game = GameState::from_board(board);

        // Rook move increments clock
        game.apply_move(make_quiet_move((2, -5), (2, -3)));
        assert_eq!(game.board.halfmove_clock, 1);

        // Black rook move increments clock
        game.apply_move(make_quiet_move((-2, 5), (-2, 3)));
        assert_eq!(game.board.halfmove_clock, 2);

        // Pawn move resets clock
        game.apply_move(make_quiet_move((0, -1), (0, 0)));
        assert_eq!(game.board.halfmove_clock, 0);
    }

    // ----- test: fullmove number increments after black -----

    #[test]
    fn test_fullmove_number() {
        let mut board = kings_only_board();
        board.set(
            HexCoord::new(2, -5),
            Some(Piece {
                kind: PieceKind::Rook,
                color: Color::White,
            }),
        );
        board.set(
            HexCoord::new(-2, 5),
            Some(Piece {
                kind: PieceKind::Rook,
                color: Color::Black,
            }),
        );
        let mut game = GameState::from_board(board);

        assert_eq!(game.board.fullmove_number, 1);
        game.apply_move(make_quiet_move((2, -5), (2, -3))); // white
        assert_eq!(game.board.fullmove_number, 1);
        game.apply_move(make_quiet_move((-2, 5), (-2, 3))); // black
        assert_eq!(game.board.fullmove_number, 2);
    }

    // ----- test: king position cache update -----

    #[test]
    fn test_king_position_cache_update() {
        let mut board = kings_only_board();
        board.set(
            HexCoord::new(-2, 5),
            Some(Piece {
                kind: PieceKind::Rook,
                color: Color::Black,
            }),
        );
        let mut game = GameState::from_board(board);

        // Move white king
        game.apply_move(make_quiet_move((0, -4), (0, -3)));
        assert_eq!(game.board.white_king, HexCoord::new(0, -3));

        // Move black rook (so we can check king didn't change)
        game.apply_move(make_quiet_move((-2, 5), (-2, 3)));
        assert_eq!(game.board.black_king, HexCoord::new(0, 4));

        // Undo black rook
        game.undo_move();
        // Undo white king move
        game.undo_move();
        assert_eq!(game.board.white_king, HexCoord::new(0, -4));
    }

    // ----- test: outcome_value -----

    #[test]
    fn test_outcome_value() {
        // K vs K is DrawByInsufficientMaterial => outcome 0.0
        let board = kings_only_board();
        let game = GameState::from_board(board);
        assert_eq!(game.outcome_value(), Some(0.0));

        // Starting position is Ongoing => None
        let game2 = GameState::new();
        assert_eq!(game2.outcome_value(), None);
    }
}
