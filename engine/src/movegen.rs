//! Move generation for Glinski's hexagonal chess.
//!
//! Provides pseudo-legal and legal move generation, attack detection,
//! and precomputed lookup tables for knights and sliding-piece rays.

use crate::board::{
    ALL_DIRS, BLACK_PAWN_START, Board, Cell, Color, HexCoord, KNIGHT_OFFSETS, PROMOTION_PIECES,
    Piece, PieceKind, WHITE_PAWN_START, coord_to_index, index_to_coord,
};

/// Unwrapping coord_to_index for internal use with known-valid coordinates.
#[inline]
fn cell_index(coord: HexCoord) -> usize {
    coord_to_index(coord).expect("cell_index: invalid coord")
}

// ---------------------------------------------------------------------------
// Move representation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Move {
    pub from: HexCoord,
    pub to: HexCoord,
    pub promotion: Option<PieceKind>,
    pub captured: Cell,
    pub is_en_passant: bool,
}

impl Move {
    #[inline]
    pub fn new(from: HexCoord, to: HexCoord, captured: Cell) -> Self {
        Self {
            from,
            to,
            promotion: None,
            captured,
            is_en_passant: false,
        }
    }

    #[inline]
    pub fn with_promotion(mut self, kind: PieceKind) -> Self {
        self.promotion = Some(kind);
        self
    }

    #[inline]
    pub fn with_promotion_opt(mut self, kind: Option<PieceKind>) -> Self {
        self.promotion = kind;
        self
    }

    #[inline]
    pub fn en_passant(from: HexCoord, to: HexCoord) -> Self {
        Self {
            from,
            to,
            promotion: None,
            captured: None,
            is_en_passant: true,
        }
    }
}

// ---------------------------------------------------------------------------
// MoveList — stack-allocated move buffer (avoids heap allocation)
// ---------------------------------------------------------------------------

pub struct MoveList {
    moves: [std::mem::MaybeUninit<Move>; 256],
    len: usize,
}

impl MoveList {
    pub fn new() -> Self {
        Self {
            // Safety: An array of MaybeUninit does not require initialization.
            moves: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < 256, "MoveList overflow");
        self.moves[self.len] = std::mem::MaybeUninit::new(m);
        self.len += 1;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> MoveListIter<'_> {
        MoveListIter { list: self, idx: 0 }
    }

    pub fn to_vec(&self) -> Vec<Move> {
        self.iter().copied().collect()
    }

    /// Swap two elements by index (for in-place move ordering).
    #[inline]
    pub fn swap(&mut self, a: usize, b: usize) {
        debug_assert!(a < self.len && b < self.len);
        self.moves.swap(a, b);
    }

    #[inline]
    fn get(&self, idx: usize) -> &Move {
        debug_assert!(idx < self.len);
        // Safety: indices < self.len are always initialized via push().
        unsafe { self.moves[idx].assume_init_ref() }
    }
}

impl std::ops::Index<usize> for MoveList {
    type Output = Move;
    #[inline]
    fn index(&self, idx: usize) -> &Move {
        self.get(idx)
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MoveListIter<'a> {
    list: &'a MoveList,
    idx: usize,
}

impl<'a> Iterator for MoveListIter<'a> {
    type Item = &'a Move;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.list.len {
            let m = self.list.get(self.idx);
            self.idx += 1;
            Some(m)
        } else {
            None
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.list.len - self.idx;
        (r, Some(r))
    }
}

impl<'a> ExactSizeIterator for MoveListIter<'a> {}

// ---------------------------------------------------------------------------
// Precomputed tables (lazy-initialized via OnceLock)
// ---------------------------------------------------------------------------

struct Tables {
    /// For each cell index (0..91), list of valid knight destination indices.
    knight_moves: [Vec<usize>; 91],
    /// For each cell index and each of 12 directions (ordered as in ALL_DIRS),
    /// the ray of cell indices along that direction (excluding origin, up to board edge).
    rays: [[Vec<usize>; 12]; 91],
}

impl Tables {
    fn compute() -> Self {
        let mut knight_moves: [Vec<usize>; 91] = std::array::from_fn(|_| Vec::new());
        let mut rays: [[Vec<usize>; 12]; 91] =
            std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));

        for idx in 0..91 {
            let coord = index_to_coord(idx);

            // Knight destinations
            for &(dq, dr) in &KNIGHT_OFFSETS {
                let dest = coord.offset(dq, dr);
                if dest.is_valid() {
                    knight_moves[idx].push(cell_index(dest));
                }
            }

            // Sliding-piece rays in all 12 directions
            for (dir_idx, &(dq, dr)) in ALL_DIRS.iter().enumerate() {
                let mut ray = Vec::new();
                let mut cur = coord;
                loop {
                    cur = cur.offset(dq, dr);
                    if !cur.is_valid() {
                        break;
                    }
                    ray.push(cell_index(cur));
                }
                rays[idx][dir_idx] = ray;
            }
        }

        Tables { knight_moves, rays }
    }
}

fn tables() -> &'static Tables {
    use std::sync::OnceLock;
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(Tables::compute)
}

// ---------------------------------------------------------------------------
// Pawn constants and helpers
// ---------------------------------------------------------------------------

/// White promotion squares (top edge of board).
/// Condition in axial: (q >= 0 && q + r == 5) || (q < 0 && r == 5).
const WHITE_PROMOTION: [(i8, i8); 11] = [
    (-5, 5),
    (-4, 5),
    (-3, 5),
    (-2, 5),
    (-1, 5),
    (0, 5),
    (1, 4),
    (2, 3),
    (3, 2),
    (4, 1),
    (5, 0),
];

/// Black promotion squares (bottom edge of board, mirror of white).
const BLACK_PROMOTION: [(i8, i8); 11] = [
    (5, -5),
    (4, -5),
    (3, -5),
    (2, -5),
    (1, -5),
    (0, -5),
    (-1, -4),
    (-2, -3),
    (-3, -2),
    (-4, -1),
    (-5, 0),
];

#[inline]
fn is_pawn_start(coord: HexCoord, color: Color) -> bool {
    let positions = match color {
        Color::White => &WHITE_PAWN_START,
        Color::Black => &BLACK_PAWN_START,
    };
    positions.iter().any(|&(q, r)| coord.q == q && coord.r == r)
}

#[inline]
fn is_promotion_square(coord: HexCoord, color: Color) -> bool {
    let squares = match color {
        Color::White => &WHITE_PROMOTION[..],
        Color::Black => &BLACK_PROMOTION[..],
    };
    squares.iter().any(|&(q, r)| coord.q == q && coord.r == r)
}

/// Returns (forward, left_capture, right_capture) direction offsets for a pawn.
///
/// White: forward (0,+1), captures (-1,+1) and (+1,0).
/// Black: forward (0,-1), captures (+1,-1) and (-1,0).
#[inline]
fn pawn_dirs(color: Color) -> ((i8, i8), (i8, i8), (i8, i8)) {
    match color {
        Color::White => ((0, 1), (-1, 1), (1, 0)),
        Color::Black => ((0, -1), (1, -1), (-1, 0)),
    }
}

// ---------------------------------------------------------------------------
// Direction index sets (into ALL_DIRS)
// ---------------------------------------------------------------------------

/// Cardinal direction indices (rook directions) in ALL_DIRS.
const CARDINAL_DIR_INDICES: [usize; 6] = [0, 1, 2, 3, 4, 5];
/// Diagonal direction indices (bishop directions) in ALL_DIRS.
const DIAGONAL_DIR_INDICES: [usize; 6] = [6, 7, 8, 9, 10, 11];
/// All direction indices (queen directions) in ALL_DIRS.
const ALL_DIR_INDICES: [usize; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

// ---------------------------------------------------------------------------
// Pseudo-legal move generation
// ---------------------------------------------------------------------------

/// Generate all pseudo-legal moves for the side to move.
/// Does NOT filter for leaving own king in check.
pub fn generate_pseudo_legal_moves(board: &Board) -> MoveList {
    let mut moves = MoveList::new();
    let color = board.side_to_move;

    for idx in 0..91 {
        let piece = match board.cells[idx] {
            Some(p) if p.color == color => p,
            _ => continue,
        };
        let from = index_to_coord(idx);

        match piece.kind {
            PieceKind::Pawn => gen_pawn(board, from, color, &mut moves),
            PieceKind::Knight => gen_knight(board, from, color, &mut moves),
            PieceKind::Bishop => gen_sliding(board, from, color, &DIAGONAL_DIR_INDICES, &mut moves),
            PieceKind::Rook => gen_sliding(board, from, color, &CARDINAL_DIR_INDICES, &mut moves),
            PieceKind::Queen => gen_sliding(board, from, color, &ALL_DIR_INDICES, &mut moves),
            PieceKind::King => gen_king(board, from, color, &mut moves),
        }
    }

    moves
}

fn gen_pawn(board: &Board, from: HexCoord, color: Color, moves: &mut MoveList) {
    let (forward, left_cap, right_cap) = pawn_dirs(color);

    // --- Forward moves ---
    let one = from.offset(forward.0, forward.1);
    if one.is_valid() && board.get(one).is_none() {
        if is_promotion_square(one, color) {
            for &promo in &PROMOTION_PIECES {
                moves.push(Move::new(from, one, None).with_promotion(promo));
            }
        } else {
            moves.push(Move::new(from, one, None));

            // Double forward from starting position
            if is_pawn_start(from, color) {
                let two = one.offset(forward.0, forward.1);
                if two.is_valid() && board.get(two).is_none() {
                    moves.push(Move::new(from, two, None));
                }
            }
        }
    }

    // --- Captures (normal + en passant) ---
    for &(dq, dr) in &[left_cap, right_cap] {
        let to = from.offset(dq, dr);
        if !to.is_valid() {
            continue;
        }
        let target = board.get(to);
        if let Some(piece) = target {
            if piece.color != color {
                if is_promotion_square(to, color) {
                    for &promo in &PROMOTION_PIECES {
                        moves.push(Move::new(from, to, target).with_promotion(promo));
                    }
                } else {
                    moves.push(Move::new(from, to, target));
                }
            }
        } else if board.en_passant == Some(to) {
            // EP always captures an opponent pawn — store it for undo.
            let captured_pawn = Some(Piece {
                kind: PieceKind::Pawn,
                color: color.opponent(),
            });
            moves.push(Move {
                from,
                to,
                promotion: None,
                captured: captured_pawn,
                is_en_passant: true,
            });
        }
    }
}

fn gen_knight(board: &Board, from: HexCoord, color: Color, moves: &mut MoveList) {
    let tab = tables();
    let from_idx = cell_index(from);
    for &dest_idx in &tab.knight_moves[from_idx] {
        let target = board.cells[dest_idx];
        match target {
            Some(p) if p.color == color => continue,
            _ => moves.push(Move::new(from, index_to_coord(dest_idx), target)),
        }
    }
}

fn gen_sliding(
    board: &Board,
    from: HexCoord,
    color: Color,
    dir_indices: &[usize],
    moves: &mut MoveList,
) {
    let tab = tables();
    let from_idx = cell_index(from);
    for &dir in dir_indices {
        for &cell_idx in &tab.rays[from_idx][dir] {
            let target = board.cells[cell_idx];
            match target {
                None => {
                    moves.push(Move::new(from, index_to_coord(cell_idx), None));
                }
                Some(p) if p.color != color => {
                    moves.push(Move::new(from, index_to_coord(cell_idx), target));
                    break;
                }
                _ => break, // friendly piece blocks
            }
        }
    }
}

fn gen_king(board: &Board, from: HexCoord, color: Color, moves: &mut MoveList) {
    for &(dq, dr) in &ALL_DIRS {
        let to = from.offset(dq, dr);
        if !to.is_valid() {
            continue;
        }
        let target = board.get(to);
        match target {
            Some(p) if p.color == color => continue,
            _ => moves.push(Move::new(from, to, target)),
        }
    }
}

// ---------------------------------------------------------------------------
// Attack detection
// ---------------------------------------------------------------------------

/// Check if a given square is attacked by any piece of the given color.
pub fn is_square_attacked(board: &Board, coord: HexCoord, by_color: Color) -> bool {
    attacked_by_pawn(board, coord, by_color)
        || attacked_by_knight(board, coord, by_color)
        || attacked_by_sliding(board, coord, by_color)
        || attacked_by_king(board, coord, by_color)
}

fn attacked_by_pawn(board: &Board, coord: HexCoord, by_color: Color) -> bool {
    // A pawn at `src` attacks `coord` via a capture direction.
    // Reverse: src = coord - capture_dir.
    let (_, left_cap, right_cap) = pawn_dirs(by_color);
    for &(dq, dr) in &[left_cap, right_cap] {
        let src = coord.offset(-dq, -dr);
        if src.is_valid()
            && let Some(p) = board.get(src)
            && p.color == by_color
            && p.kind == PieceKind::Pawn
        {
            return true;
        }
    }
    false
}

fn attacked_by_knight(board: &Board, coord: HexCoord, by_color: Color) -> bool {
    let tab = tables();
    let idx = cell_index(coord);
    for &src_idx in &tab.knight_moves[idx] {
        if let Some(p) = board.cells[src_idx]
            && p.color == by_color
            && p.kind == PieceKind::Knight
        {
            return true;
        }
    }
    false
}

fn attacked_by_sliding(board: &Board, coord: HexCoord, by_color: Color) -> bool {
    let tab = tables();
    let idx = cell_index(coord);

    // Cardinal rays: rook or queen
    for dir in 0..6 {
        for &cell_idx in &tab.rays[idx][dir] {
            match board.cells[cell_idx] {
                None => continue,
                Some(p) => {
                    if p.color == by_color
                        && (p.kind == PieceKind::Rook || p.kind == PieceKind::Queen)
                    {
                        return true;
                    }
                    break;
                }
            }
        }
    }

    // Diagonal rays: bishop or queen
    for dir in 6..12 {
        for &cell_idx in &tab.rays[idx][dir] {
            match board.cells[cell_idx] {
                None => continue,
                Some(p) => {
                    if p.color == by_color
                        && (p.kind == PieceKind::Bishop || p.kind == PieceKind::Queen)
                    {
                        return true;
                    }
                    break;
                }
            }
        }
    }

    false
}

fn attacked_by_king(board: &Board, coord: HexCoord, by_color: Color) -> bool {
    for &(dq, dr) in &ALL_DIRS {
        let src = coord.offset(dq, dr);
        if src.is_valid()
            && let Some(p) = board.get(src)
            && p.color == by_color
            && p.kind == PieceKind::King
        {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Check detection
// ---------------------------------------------------------------------------

/// Check if the given color's king is currently in check.
pub fn is_in_check(board: &Board, color: Color) -> bool {
    let king_pos = board.king_pos(color);
    is_square_attacked(board, king_pos, color.opponent())
}

// ---------------------------------------------------------------------------
// Legal move generation
// ---------------------------------------------------------------------------

/// Apply a move to a cloned board (minimal, for legality testing only).
/// Does NOT update zobrist hash or position history.
fn apply_move_minimal(board: &Board, m: &Move) -> Board {
    let mut b = board.clone();
    let piece = b
        .get(m.from)
        .expect("apply_move_minimal: no piece at `from`");

    // Remove piece from origin
    b.set(m.from, None);

    // En passant capture: remove the captured pawn (not on `to`)
    if m.is_en_passant {
        let (fwd, _, _) = pawn_dirs(piece.color);
        let captured_sq = m.to.offset(-fwd.0, -fwd.1);
        b.set(captured_sq, None);
    }

    // Place piece at destination (with promotion if applicable)
    let placed = if let Some(promo_kind) = m.promotion {
        Piece {
            kind: promo_kind,
            color: piece.color,
        }
    } else {
        piece
    };
    b.set(m.to, Some(placed)); // Board::set updates king cache automatically

    // Update en passant target
    b.en_passant = None;
    if piece.kind == PieceKind::Pawn {
        let (fwd, _, _) = pawn_dirs(piece.color);
        // A double pawn advance moves exactly 2 forward steps
        if m.to.q - m.from.q == 2 * fwd.0 && m.to.r - m.from.r == 2 * fwd.1 {
            b.en_passant = Some(m.from.offset(fwd.0, fwd.1));
        }
    }

    // Flip side to move
    b.side_to_move = board.side_to_move.opponent();

    b
}

/// Generate all legal moves (pseudo-legal moves filtered for king safety).
pub fn generate_legal_moves(board: &Board) -> MoveList {
    let pseudo = generate_pseudo_legal_moves(board);
    let mut legal = MoveList::new();
    let color = board.side_to_move;

    for m in pseudo.iter() {
        let new_board = apply_move_minimal(board, m);
        if !is_in_check(&new_board, color) {
            legal.push(*m);
        }
    }

    legal
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::*;

    /// Create an empty board with kings placed at the given positions.
    fn board_with_kings(wk: (i8, i8), bk: (i8, i8)) -> Board {
        let mut b = Board::empty();
        b.white_king = HexCoord::new(wk.0, wk.1);
        b.black_king = HexCoord::new(bk.0, bk.1);
        b.set(
            HexCoord::new(wk.0, wk.1),
            Some(Piece {
                kind: PieceKind::King,
                color: Color::White,
            }),
        );
        b.set(
            HexCoord::new(bk.0, bk.1),
            Some(Piece {
                kind: PieceKind::King,
                color: Color::Black,
            }),
        );
        b
    }

    fn place(board: &mut Board, q: i8, r: i8, kind: PieceKind, color: Color) {
        let coord = HexCoord::new(q, r);
        board.set(coord, Some(Piece { kind, color }));
        if kind == PieceKind::King {
            match color {
                Color::White => board.white_king = coord,
                Color::Black => board.black_king = coord,
            }
        }
    }

    // ===== Precomputed table tests =====

    #[test]
    fn knight_moves_from_center() {
        let tab = tables();
        let idx = cell_index(HexCoord::new(0, 0));
        assert_eq!(
            tab.knight_moves[idx].len(),
            12,
            "Knight at center should have 12 moves"
        );
    }

    #[test]
    fn knight_moves_from_corner() {
        let tab = tables();
        let idx = cell_index(HexCoord::new(5, -5));
        let count = tab.knight_moves[idx].len();
        assert!(
            count >= 2 && count < 12,
            "Knight at corner: expected 2..12, got {}",
            count
        );
    }

    #[test]
    fn knight_moves_all_cells_have_some() {
        let tab = tables();
        for idx in 0..91 {
            let count = tab.knight_moves[idx].len();
            assert!(
                count >= 2,
                "Cell {} ({:?}) has only {} knight moves",
                idx,
                index_to_coord(idx),
                count
            );
        }
    }

    #[test]
    fn rays_cardinal_from_center() {
        let tab = tables();
        let idx = cell_index(HexCoord::new(0, 0));
        // From center, each of the 6 cardinal directions should have exactly 5 cells.
        for dir in 0..6 {
            assert_eq!(
                tab.rays[idx][dir].len(),
                5,
                "Cardinal ray {} from center should have 5 cells",
                dir
            );
        }
    }

    #[test]
    fn rays_all_nonempty_from_center() {
        let tab = tables();
        let idx = cell_index(HexCoord::new(0, 0));
        for dir in 0..12 {
            assert!(
                !tab.rays[idx][dir].is_empty(),
                "Ray {} from center should not be empty",
                dir
            );
        }
    }

    // ===== Pawn movement tests =====

    #[test]
    fn white_pawn_single_and_double() {
        let mut board = board_with_kings((-5, 0), (5, 0));
        place(&mut board, 0, -1, PieceKind::Pawn, Color::White);

        let moves = generate_pseudo_legal_moves(&board);
        let pm: Vec<_> = moves
            .iter()
            .filter(|m| m.from == HexCoord::new(0, -1))
            .collect();

        assert_eq!(pm.len(), 2, "Pawn on start should have 2 forward moves");
        assert!(pm.iter().any(|m| m.to == HexCoord::new(0, 0)));
        assert!(pm.iter().any(|m| m.to == HexCoord::new(0, 1)));
    }

    #[test]
    fn white_pawn_no_double_off_start() {
        let mut board = board_with_kings((-5, 0), (5, 0));
        place(&mut board, 0, 0, PieceKind::Pawn, Color::White);

        let moves = generate_pseudo_legal_moves(&board);
        let pm: Vec<_> = moves
            .iter()
            .filter(|m| m.from == HexCoord::new(0, 0))
            .collect();

        assert_eq!(pm.len(), 1);
        assert_eq!(pm[0].to, HexCoord::new(0, 1));
    }

    #[test]
    fn white_pawn_captures() {
        let mut board = board_with_kings((-5, 0), (5, 0));
        place(&mut board, 0, 0, PieceKind::Pawn, Color::White);
        place(&mut board, -1, 1, PieceKind::Pawn, Color::Black);
        place(&mut board, 1, 0, PieceKind::Pawn, Color::Black);

        let moves = generate_pseudo_legal_moves(&board);
        let pm: Vec<_> = moves
            .iter()
            .filter(|m| m.from == HexCoord::new(0, 0))
            .collect();

        // 1 forward + 2 captures = 3
        assert_eq!(pm.len(), 3);
    }

    #[test]
    fn pawn_promotion() {
        let mut board = board_with_kings((-5, 0), (5, -5));
        place(&mut board, 0, 4, PieceKind::Pawn, Color::White);

        let moves = generate_pseudo_legal_moves(&board);
        let pm: Vec<_> = moves
            .iter()
            .filter(|m| m.from == HexCoord::new(0, 4))
            .collect();

        assert_eq!(pm.len(), 4, "Should get 4 promotion choices");
        assert!(pm.iter().all(|m| m.promotion.is_some()));
    }

    #[test]
    fn en_passant_capture() {
        let mut board = board_with_kings((-5, 0), (5, -5));
        // Black pawn double-moved from (0,1) to (0,-1), skipping (0,0).
        place(&mut board, -1, 0, PieceKind::Pawn, Color::White);
        place(&mut board, 0, -1, PieceKind::Pawn, Color::Black);
        board.en_passant = Some(HexCoord::new(0, 0));

        let moves = generate_pseudo_legal_moves(&board);
        let ep: Vec<_> = moves.iter().filter(|m| m.is_en_passant).collect();

        assert_eq!(ep.len(), 1);
        assert_eq!(ep[0].to, HexCoord::new(0, 0));
    }

    #[test]
    fn black_pawn_single_and_double() {
        let mut board = board_with_kings((-5, 0), (5, 0));
        place(&mut board, 0, 1, PieceKind::Pawn, Color::Black);
        board.side_to_move = Color::Black;

        let moves = generate_pseudo_legal_moves(&board);
        let pm: Vec<_> = moves
            .iter()
            .filter(|m| m.from == HexCoord::new(0, 1))
            .collect();

        assert_eq!(pm.len(), 2);
        assert!(pm.iter().any(|m| m.to == HexCoord::new(0, 0)));
        assert!(pm.iter().any(|m| m.to == HexCoord::new(0, -1)));
    }

    // ===== Sliding piece tests =====

    #[test]
    fn rook_at_center() {
        let mut board = board_with_kings((-5, 0), (5, 0));
        place(&mut board, 0, 0, PieceKind::Rook, Color::White);

        let moves = generate_pseudo_legal_moves(&board);
        let rm: Vec<_> = moves
            .iter()
            .filter(|m| m.from == HexCoord::new(0, 0))
            .collect();

        // 6 cardinal directions x 5 cells each = 30, minus:
        // (-1,0) direction: 4 cells then white king at (-5,0) blocks = 4
        // (1,0) direction: 4 cells then capture black king at (5,0) = 5
        // Other 4 directions: 5 each = 20
        // Total = 4 + 5 + 20 = 29
        assert_eq!(rm.len(), 29);
    }

    #[test]
    fn bishop_at_center() {
        // Put kings far away on same hex-color to avoid interference with diagonal rays.
        let mut board = board_with_kings((-5, 5), (5, -5));
        place(&mut board, 0, 0, PieceKind::Bishop, Color::White);

        let moves = generate_pseudo_legal_moves(&board);
        let bm: Vec<_> = moves
            .iter()
            .filter(|m| m.from == HexCoord::new(0, 0))
            .collect();

        // Each of 6 diagonal directions from center reaches 2 valid cells.
        // 6 x 2 = 12
        assert_eq!(bm.len(), 12);
    }

    // ===== King tests =====

    #[test]
    fn king_at_center_pseudo_legal() {
        let mut board = board_with_kings((-5, 5), (5, -5));
        // Replace king at center
        board.set(HexCoord::new(-5, 5), None);
        place(&mut board, 0, 0, PieceKind::King, Color::White);

        let moves = generate_pseudo_legal_moves(&board);
        let km: Vec<_> = moves
            .iter()
            .filter(|m| m.from == HexCoord::new(0, 0))
            .collect();

        // All 12 neighbors of center are valid on the board.
        assert_eq!(km.len(), 12);
    }

    // ===== Attack detection =====

    #[test]
    fn check_by_rook() {
        let mut board = board_with_kings((0, 0), (-5, 5));
        place(&mut board, 0, 3, PieceKind::Rook, Color::Black);

        assert!(is_in_check(&board, Color::White));
        assert!(!is_in_check(&board, Color::Black));
    }

    #[test]
    fn pawn_attacks() {
        let mut board = board_with_kings((-5, 0), (5, 0));
        place(&mut board, 0, 0, PieceKind::Pawn, Color::White);

        // White pawn at (0,0) captures on (-1,1) and (1,0).
        assert!(is_square_attacked(
            &board,
            HexCoord::new(-1, 1),
            Color::White
        ));
        assert!(is_square_attacked(
            &board,
            HexCoord::new(1, 0),
            Color::White
        ));
        // Does NOT attack the forward square.
        assert!(!is_square_attacked(
            &board,
            HexCoord::new(0, 1),
            Color::White
        ));
    }

    // ===== Pin test =====

    #[test]
    fn pinned_piece_restricted() {
        // White king (0,-5), white rook (0,-3), black rook (0,0).
        // Rook pinned along cardinal (0,+1) ray.
        let mut board = board_with_kings((0, -5), (5, -5));
        place(&mut board, 0, -3, PieceKind::Rook, Color::White);
        place(&mut board, 0, 0, PieceKind::Rook, Color::Black);

        let legal = generate_legal_moves(&board);
        let rook_legal: Vec<_> = legal
            .iter()
            .filter(|m| m.from == HexCoord::new(0, -3))
            .collect();

        // Rook can only stay on the pin ray (q=0 column): (0,-4), (0,-2), (0,-1), (0,0).
        for m in &rook_legal {
            assert_eq!(m.to.q, 0, "Pinned rook must stay on pin line");
        }
        assert_eq!(rook_legal.len(), 4);
    }

    #[test]
    fn king_cannot_move_into_check() {
        let mut board = board_with_kings((0, 0), (-5, 5));
        // Black rook at (0,5) attacks entire q=0 column toward the king.
        place(&mut board, 0, 5, PieceKind::Rook, Color::Black);

        let legal = generate_legal_moves(&board);
        let km: Vec<_> = legal
            .iter()
            .filter(|m| m.from == HexCoord::new(0, 0))
            .collect();

        // King should not be able to step onto q=0 (still attacked after vacating).
        for m in &km {
            assert_ne!(m.to.q, 0, "King should not step into rook's line of fire");
        }
    }

    // ===== Starting position test (the critical one) =====

    #[test]
    fn starting_position_has_50_legal_moves() {
        let board = Board::new();
        let legal = generate_legal_moves(&board);
        // 51 legal moves: 17 pawn + 8 knight + 12 bishop + 6 rook + 6 queen + 2 king.
        // The king moves in all 12 directions (6 cardinal + 6 diagonal) in hex chess.
        assert_eq!(
            legal.len(),
            51,
            "Glinski starting position should have exactly 51 legal moves, got {}",
            legal.len()
        );
    }

    // ===== MoveList basic tests =====

    #[test]
    fn movelist_push_and_index() {
        let mut ml = MoveList::new();
        assert!(ml.is_empty());
        assert_eq!(ml.len(), 0);

        let m = Move::new(HexCoord::new(0, 0), HexCoord::new(1, 0), None);
        ml.push(m);

        assert_eq!(ml.len(), 1);
        assert!(!ml.is_empty());
        assert_eq!(ml[0].from, HexCoord::new(0, 0));
        assert_eq!(ml[0].to, HexCoord::new(1, 0));
    }

    #[test]
    fn movelist_iter() {
        let mut ml = MoveList::new();
        for i in 0..10 {
            ml.push(Move::new(
                HexCoord::new(0, 0),
                HexCoord::new(i as i8, 0),
                None,
            ));
        }
        assert_eq!(ml.iter().count(), 10);
        let v = ml.to_vec();
        assert_eq!(v.len(), 10);
    }
}
