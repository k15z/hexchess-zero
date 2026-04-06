use std::fmt;
use std::sync::LazyLock;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of valid cells on the Glinski hex board.
pub const NUM_CELLS: usize = 91;

/// Radius of the hex board (distance from center to edge).
pub const BOARD_RADIUS: i8 = 5;

// ---------------------------------------------------------------------------
// HexCoord
// ---------------------------------------------------------------------------

/// Axial coordinate on the hex grid.
///
/// Mapping from doubled coordinates `(x, y)`:  `q = x`, `r = (y - x) / 2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HexCoord {
    pub q: i8,
    pub r: i8,
}

impl HexCoord {
    #[inline]
    pub const fn new(q: i8, r: i8) -> Self {
        Self { q, r }
    }

    /// Returns true if this coordinate lies on the 91-cell Glinski board.
    ///
    /// In cube coordinates the constraint is `max(|q|, |r|, |s|) <= 5` where `s = -q - r`.
    #[inline]
    pub const fn is_valid(&self) -> bool {
        let s = -self.q - self.r;
        let aq = if self.q < 0 { -self.q } else { self.q };
        let ar = if self.r < 0 { -self.r } else { self.r };
        let a_s = if s < 0 { -s } else { s };
        // max of three absolutes
        let m1 = if aq > ar { aq } else { ar };
        let m = if m1 > a_s { m1 } else { a_s };
        m <= BOARD_RADIUS
    }

    /// Offset by a direction vector; returns `Some` if the result is valid.
    #[inline]
    pub fn step(self, dq: i8, dr: i8) -> Option<Self> {
        let c = Self::new(self.q + dq, self.r + dr);
        if c.is_valid() { Some(c) } else { None }
    }

    /// Offset by a direction vector without validity check.
    #[inline]
    pub const fn offset(self, dq: i8, dr: i8) -> Self {
        Self::new(self.q + dq, self.r + dr)
    }

    /// Glinski human-readable notation for this cell, e.g. `f6` (center) or
    /// `g1` (white king starting square). Returns `None` for off-board
    /// coordinates. See the docs `rules` page for the full mapping.
    pub fn to_notation(&self) -> Option<String> {
        if !self.is_valid() {
            return None;
        }
        const FILES: [char; 11] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l'];
        let file = FILES[(self.q + 5) as usize];
        let rank = self.r as i16 + 6 + (self.q as i16).min(0);
        Some(format!("{}{}", file, rank))
    }

    /// Parse Glinski notation like `f6` or `g1` back into axial coordinates.
    /// Returns `None` if the string is malformed or off-board.
    pub fn from_notation(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() < 2 {
            return None;
        }
        let file_char = bytes[0] as char;
        let q: i8 = match file_char {
            'a' => -5,
            'b' => -4,
            'c' => -3,
            'd' => -2,
            'e' => -1,
            'f' => 0,
            'g' => 1,
            'h' => 2,
            'i' => 3,
            'k' => 4,
            'l' => 5,
            _ => return None,
        };
        let rank: i16 = std::str::from_utf8(&bytes[1..]).ok()?.parse().ok()?;
        if !(1..=11).contains(&rank) {
            return None;
        }
        let r = rank - 6 - (q as i16).min(0);
        if !(-5..=5).contains(&r) {
            return None;
        }
        let coord = Self::new(q, r as i8);
        if coord.is_valid() { Some(coord) } else { None }
    }
}

impl fmt::Display for HexCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.q, self.r)
    }
}

// ---------------------------------------------------------------------------
// Cell indexing  (precomputed lookup tables)
// ---------------------------------------------------------------------------

/// Ordered list of all 91 valid axial coordinates, sorted by (q, r).
/// Index in this array == cell index used for board storage.
pub const ALL_COORDS: [HexCoord; NUM_CELLS] = {
    let mut arr = [HexCoord { q: 0, r: 0 }; NUM_CELLS];
    let mut idx = 0usize;
    let mut q: i8 = -5;
    while q <= 5 {
        let mut r: i8 = -5;
        while r <= 5 {
            let c = HexCoord { q, r };
            if c.is_valid() {
                arr[idx] = c;
                idx += 1;
            }
            r += 1;
        }
        q += 1;
    }
    arr
};

/// Lookup table: for axial `(q, r)` mapped to `[q+5][r+5]`, stores the cell
/// index (0..90) or `u8::MAX` if the coordinate is off-board.
const COORD_TO_INDEX_TABLE: [[u8; 11]; 11] = {
    let mut table = [[u8::MAX; 11]; 11];
    let mut i = 0usize;
    while i < NUM_CELLS {
        let c = ALL_COORDS[i];
        table[(c.q + 5) as usize][(c.r + 5) as usize] = i as u8;
        i += 1;
    }
    table
};

/// Convert an axial coordinate to its cell index (0..90).
/// Returns `None` for off-board coordinates.
#[inline]
pub fn coord_to_index(coord: HexCoord) -> Option<usize> {
    let qi = (coord.q + 5) as usize;
    let ri = (coord.r + 5) as usize;
    if qi < 11 && ri < 11 {
        let v = COORD_TO_INDEX_TABLE[qi][ri];
        if v != u8::MAX { Some(v as usize) } else { None }
    } else {
        None
    }
}

/// Convert a cell index (0..90) back to its axial coordinate.
/// Panics if `index >= NUM_CELLS`.
#[inline]
pub fn index_to_coord(index: usize) -> HexCoord {
    ALL_COORDS[index]
}

/// Iterator over all 91 valid coordinates with their indices.
pub fn all_coords() -> impl Iterator<Item = (usize, HexCoord)> {
    ALL_COORDS.iter().copied().enumerate()
}

// ---------------------------------------------------------------------------
// Hex directions
// ---------------------------------------------------------------------------

/// The 6 edge-adjacent (cardinal) directions in axial coordinates.
pub const CARDINAL_DIRS: [(i8, i8); 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)];

/// The 6 vertex-adjacent (diagonal) directions in axial coordinates.
pub const DIAGONAL_DIRS: [(i8, i8); 6] = [(2, -1), (-2, 1), (1, 1), (-1, -1), (1, -2), (-1, 2)];

/// All 12 directions (6 cardinal + 6 diagonal).
pub const ALL_DIRS: [(i8, i8); 12] = [
    // cardinal
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, -1),
    (-1, 1),
    // diagonal
    (2, -1),
    (-2, 1),
    (1, 1),
    (-1, -1),
    (1, -2),
    (-1, 2),
];

/// Hex knight jump offsets in axial coordinates (12 jumps).
pub const KNIGHT_OFFSETS: [(i8, i8); 12] = [
    (1, 2),
    (2, 1),
    (3, -1),
    (1, -3),
    (-1, -2),
    (-2, -1),
    (-3, 1),
    (-1, 3),
    (-2, 3),
    (-3, 2),
    (2, -3),
    (3, -2),
];

/// White pawn starting positions (axial coords).
pub const WHITE_PAWN_START: [(i8, i8); 9] = [
    (-4, -1),
    (-3, -1),
    (-2, -1),
    (-1, -1),
    (0, -1),
    (1, -2),
    (2, -3),
    (3, -4),
    (4, -5),
];

/// Black pawn starting positions (mirror of white).
pub const BLACK_PAWN_START: [(i8, i8); 9] = [
    (4, 1),
    (3, 1),
    (2, 1),
    (1, 1),
    (0, 1),
    (-1, 2),
    (-2, 3),
    (-3, 4),
    (-4, 5),
];

// ---------------------------------------------------------------------------
// Pieces
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Color {
    White,
    Black,
}

impl Color {
    #[inline]
    pub const fn opponent(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    /// Index for array lookups (White = 0, Black = 1).
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            Color::White => 0,
            Color::Black => 1,
        }
    }
}

impl Color {
    /// Lowercase name for FFI/serialization.
    pub const fn as_str(self) -> &'static str {
        match self {
            Color::White => "white",
            Color::Black => "black",
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceKind {
    pub const COUNT: usize = 6;

    #[inline]
    pub const fn index(self) -> usize {
        match self {
            PieceKind::Pawn => 0,
            PieceKind::Knight => 1,
            PieceKind::Bishop => 2,
            PieceKind::Rook => 3,
            PieceKind::Queen => 4,
            PieceKind::King => 5,
        }
    }

    /// Single-character symbol (uppercase).
    pub const fn symbol(self) -> char {
        match self {
            PieceKind::Pawn => 'P',
            PieceKind::Knight => 'N',
            PieceKind::Bishop => 'B',
            PieceKind::Rook => 'R',
            PieceKind::Queen => 'Q',
            PieceKind::King => 'K',
        }
    }

    /// Lowercase name for FFI/serialization.
    pub const fn as_str(self) -> &'static str {
        match self {
            PieceKind::Pawn => "pawn",
            PieceKind::Knight => "knight",
            PieceKind::Bishop => "bishop",
            PieceKind::Rook => "rook",
            PieceKind::Queen => "queen",
            PieceKind::King => "king",
        }
    }

    /// Parse from lowercase name. Returns None for unrecognized strings.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pawn" => Some(PieceKind::Pawn),
            "knight" => Some(PieceKind::Knight),
            "bishop" => Some(PieceKind::Bishop),
            "rook" => Some(PieceKind::Rook),
            "queen" => Some(PieceKind::Queen),
            "king" => Some(PieceKind::King),
            _ => None,
        }
    }
}

/// Promotion piece choices.
pub const PROMOTION_PIECES: [PieceKind; 4] = [
    PieceKind::Queen,
    PieceKind::Rook,
    PieceKind::Bishop,
    PieceKind::Knight,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
}

impl Piece {
    #[inline]
    pub const fn new(kind: PieceKind, color: Color) -> Self {
        Self { kind, color }
    }

    /// Display character: uppercase for White, lowercase for Black.
    pub fn char(&self) -> char {
        let c = self.kind.symbol();
        match self.color {
            Color::White => c,
            Color::Black => c.to_ascii_lowercase(),
        }
    }
}

/// A cell is either empty or contains a piece.
pub type Cell = Option<Piece>;

// ---------------------------------------------------------------------------
// Zobrist hashing
// ---------------------------------------------------------------------------

pub struct ZobristKeys {
    /// Keys for pieces: [cell_index][piece_kind][color]
    pub pieces: [[[u64; 2]; PieceKind::COUNT]; NUM_CELLS],
    /// Key XOR-ed when it is Black's turn.
    pub side_to_move: u64,
    /// Keys for en-passant target cells (indexed by cell index).
    pub en_passant: [u64; NUM_CELLS],
}

impl ZobristKeys {
    fn generate() -> Self {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_CAFE_u64);
        let mut keys = ZobristKeys {
            pieces: [[[0u64; 2]; PieceKind::COUNT]; NUM_CELLS],
            side_to_move: 0,
            en_passant: [0u64; NUM_CELLS],
        };
        for cell in 0..NUM_CELLS {
            for kind in 0..PieceKind::COUNT {
                for color in 0..2 {
                    keys.pieces[cell][kind][color] = rng.random();
                }
            }
        }
        keys.side_to_move = rng.random();
        for cell in 0..NUM_CELLS {
            keys.en_passant[cell] = rng.random();
        }
        keys
    }

    /// Get the Zobrist key component for a piece on a cell.
    #[inline]
    pub fn piece_key(&self, cell_idx: usize, piece: Piece) -> u64 {
        self.pieces[cell_idx][piece.kind.index()][piece.color.index()]
    }
}

pub static ZOBRIST: LazyLock<ZobristKeys> = LazyLock::new(ZobristKeys::generate);

// ---------------------------------------------------------------------------
// Board
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Board {
    pub cells: [Cell; NUM_CELLS],
    pub side_to_move: Color,
    pub halfmove_clock: u16,
    pub fullmove_number: u16,
    pub en_passant: Option<HexCoord>,
    pub zobrist_hash: u64,
    pub white_king: HexCoord,
    pub black_king: HexCoord,
}

impl Board {
    // -- Construction -------------------------------------------------------

    /// Create an empty board (no pieces, White to move).
    pub fn empty() -> Self {
        Self {
            cells: [None; NUM_CELLS],
            side_to_move: Color::White,
            halfmove_clock: 0,
            fullmove_number: 1,
            en_passant: None,
            zobrist_hash: 0,
            white_king: HexCoord::new(0, 0), // placeholder
            black_king: HexCoord::new(0, 0), // placeholder
        }
    }

    /// Set up the standard Glinski starting position.
    pub fn new() -> Self {
        let mut board = Self::empty();

        // --- White pieces ---
        for &(q, r) in &WHITE_PAWN_START {
            board.place(
                HexCoord::new(q, r),
                Piece::new(PieceKind::Pawn, Color::White),
            );
        }

        // Rooks
        board.place(
            HexCoord::new(-3, -2),
            Piece::new(PieceKind::Rook, Color::White),
        );
        board.place(
            HexCoord::new(3, -5),
            Piece::new(PieceKind::Rook, Color::White),
        );
        // Knights
        board.place(
            HexCoord::new(-2, -3),
            Piece::new(PieceKind::Knight, Color::White),
        );
        board.place(
            HexCoord::new(2, -5),
            Piece::new(PieceKind::Knight, Color::White),
        );
        // Bishops
        board.place(
            HexCoord::new(0, -5),
            Piece::new(PieceKind::Bishop, Color::White),
        );
        board.place(
            HexCoord::new(0, -4),
            Piece::new(PieceKind::Bishop, Color::White),
        );
        board.place(
            HexCoord::new(0, -3),
            Piece::new(PieceKind::Bishop, Color::White),
        );
        // Queen
        board.place(
            HexCoord::new(-1, -4),
            Piece::new(PieceKind::Queen, Color::White),
        );
        // King
        let wk = HexCoord::new(1, -5);
        board.place(wk, Piece::new(PieceKind::King, Color::White));
        board.white_king = wk;

        // --- Black pieces ---
        for &(q, r) in &BLACK_PAWN_START {
            board.place(
                HexCoord::new(q, r),
                Piece::new(PieceKind::Pawn, Color::Black),
            );
        }

        // Rooks
        board.place(
            HexCoord::new(3, 2),
            Piece::new(PieceKind::Rook, Color::Black),
        );
        board.place(
            HexCoord::new(-3, 5),
            Piece::new(PieceKind::Rook, Color::Black),
        );
        // Knights
        board.place(
            HexCoord::new(2, 3),
            Piece::new(PieceKind::Knight, Color::Black),
        );
        board.place(
            HexCoord::new(-2, 5),
            Piece::new(PieceKind::Knight, Color::Black),
        );
        // Bishops
        board.place(
            HexCoord::new(0, 5),
            Piece::new(PieceKind::Bishop, Color::Black),
        );
        board.place(
            HexCoord::new(0, 4),
            Piece::new(PieceKind::Bishop, Color::Black),
        );
        board.place(
            HexCoord::new(0, 3),
            Piece::new(PieceKind::Bishop, Color::Black),
        );
        // Queen
        board.place(
            HexCoord::new(1, 4),
            Piece::new(PieceKind::Queen, Color::Black),
        );
        // King
        let bk = HexCoord::new(-1, 5);
        board.place(bk, Piece::new(PieceKind::King, Color::Black));
        board.black_king = bk;

        // White to move at start
        board.side_to_move = Color::White;
        // Recompute hash from scratch to include side-to-move
        board.zobrist_hash = board.compute_zobrist();

        board
    }

    /// Alias for `new()`.
    pub fn starting_position() -> Self {
        Self::new()
    }

    // -- Internal helpers ---------------------------------------------------

    /// Place a piece during setup, updating the Zobrist hash.
    fn place(&mut self, coord: HexCoord, piece: Piece) {
        let idx = coord_to_index(coord).expect("place: invalid coord");
        debug_assert!(self.cells[idx].is_none(), "place: cell already occupied");
        self.cells[idx] = Some(piece);
        self.zobrist_hash ^= ZOBRIST.piece_key(idx, piece);
    }

    /// Recompute the full Zobrist hash from scratch (used during init).
    fn compute_zobrist(&self) -> u64 {
        let mut h: u64 = 0;
        for (idx, cell) in self.cells.iter().enumerate() {
            if let Some(piece) = cell {
                h ^= ZOBRIST.piece_key(idx, *piece);
            }
        }
        if self.side_to_move == Color::Black {
            h ^= ZOBRIST.side_to_move;
        }
        if let Some(ep) = self.en_passant
            && let Some(ep_idx) = coord_to_index(ep)
        {
            h ^= ZOBRIST.en_passant[ep_idx];
        }
        h
    }

    // -- Accessors ----------------------------------------------------------

    /// Get the contents of a cell.
    #[inline]
    pub fn get(&self, coord: HexCoord) -> Cell {
        match coord_to_index(coord) {
            Some(idx) => self.cells[idx],
            None => None,
        }
    }

    /// Set a cell, updating the Zobrist hash incrementally.
    pub fn set(&mut self, coord: HexCoord, cell: Cell) {
        let idx = coord_to_index(coord).expect("set: invalid coord");

        // Remove old piece from hash
        if let Some(old) = self.cells[idx] {
            self.zobrist_hash ^= ZOBRIST.piece_key(idx, old);
        }
        // Add new piece to hash
        if let Some(new) = cell {
            self.zobrist_hash ^= ZOBRIST.piece_key(idx, new);
        }

        // Update king cache
        if let Some(piece) = cell
            && piece.kind == PieceKind::King
        {
            match piece.color {
                Color::White => self.white_king = coord,
                Color::Black => self.black_king = coord,
            }
        }

        self.cells[idx] = cell;
    }

    /// Get the position of the king for the given color.
    #[inline]
    pub fn king_pos(&self, color: Color) -> HexCoord {
        match color {
            Color::White => self.white_king,
            Color::Black => self.black_king,
        }
    }

    /// Iterate over all pieces of a given color, yielding `(HexCoord, Piece)`.
    pub fn all_pieces(&self, color: Color) -> impl Iterator<Item = (HexCoord, Piece)> + '_ {
        self.cells
            .iter()
            .enumerate()
            .filter_map(move |(idx, cell)| {
                cell.and_then(|p| {
                    if p.color == color {
                        Some((index_to_coord(idx), p))
                    } else {
                        None
                    }
                })
            })
    }

    /// Count all pieces of a given color.
    pub fn piece_count(&self, color: Color) -> usize {
        self.cells
            .iter()
            .filter(|c| c.is_some_and(|p| p.color == color))
            .count()
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print in a simple row-based format.
        // We iterate over axial coords grouped by `r` (rows), with `q` increasing left-to-right.
        writeln!(f, "  Side to move: {}", self.side_to_move)?;
        writeln!(
            f,
            "  Halfmove clock: {}, Fullmove: {}",
            self.halfmove_clock, self.fullmove_number
        )?;
        if let Some(ep) = self.en_passant {
            writeln!(f, "  En passant: {}", ep)?;
        }
        writeln!(f)?;

        // Print rows from top (r = max for each q) to bottom.
        // For a hex board, iterate r from 5 down to -5, within each r iterate q.
        for r in (-5..=5).rev() {
            // Indentation based on row to give hex shape
            let indent = ((r + 5) as i16).unsigned_abs() as usize;
            write!(f, "{:width$}", "", width = indent)?;

            let mut any = false;
            for q in -5..=5i8 {
                let coord = HexCoord::new(q, r);
                if coord.is_valid() {
                    if any {
                        write!(f, " ")?;
                    }
                    match self.get(coord) {
                        Some(piece) => write!(f, "{}", piece.char())?,
                        None => write!(f, ".")?,
                    }
                    any = true;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Glinski notation -----------------------------------------------------

    #[test]
    fn test_notation_known_squares() {
        // Center
        assert_eq!(HexCoord::new(0, 0).to_notation().unwrap(), "f6");
        // White starting pieces (Glinski standard layout)
        assert_eq!(HexCoord::new(1, -5).to_notation().unwrap(), "g1"); // king
        assert_eq!(HexCoord::new(-1, -4).to_notation().unwrap(), "e1"); // queen
        assert_eq!(HexCoord::new(-3, -2).to_notation().unwrap(), "c1"); // rook
        assert_eq!(HexCoord::new(3, -5).to_notation().unwrap(), "i1"); // rook
        assert_eq!(HexCoord::new(-2, -3).to_notation().unwrap(), "d1"); // knight
        assert_eq!(HexCoord::new(2, -5).to_notation().unwrap(), "h1"); // knight
        assert_eq!(HexCoord::new(0, -5).to_notation().unwrap(), "f1"); // bishop
        assert_eq!(HexCoord::new(0, -4).to_notation().unwrap(), "f2");
        assert_eq!(HexCoord::new(0, -3).to_notation().unwrap(), "f3");
        // Pawns form a V: b1, c2, d3, e4, f5, g4, h3, i2, k1
        assert_eq!(HexCoord::new(-4, -1).to_notation().unwrap(), "b1");
        assert_eq!(HexCoord::new(-3, -1).to_notation().unwrap(), "c2");
        assert_eq!(HexCoord::new(0, -1).to_notation().unwrap(), "f5");
        assert_eq!(HexCoord::new(2, -3).to_notation().unwrap(), "h3");
        assert_eq!(HexCoord::new(4, -5).to_notation().unwrap(), "k1");
        // Top of center file
        assert_eq!(HexCoord::new(0, 5).to_notation().unwrap(), "f11");
        // Black king starting square
        assert_eq!(HexCoord::new(-1, 5).to_notation().unwrap(), "e10");
    }

    #[test]
    fn test_notation_roundtrip_all_cells() {
        for (_, c) in all_coords() {
            let s = c.to_notation().expect("valid coord");
            let parsed = HexCoord::from_notation(&s).expect("roundtrip");
            assert_eq!(parsed, c, "{} -> {} -> {:?}", c, s, parsed);
        }
    }

    #[test]
    fn test_notation_all_unique() {
        let mut names: Vec<String> = all_coords()
            .map(|(_, c)| c.to_notation().unwrap())
            .collect();
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate notation labels");
    }

    #[test]
    fn test_notation_file_counts() {
        // Each file should have the expected number of cells.
        let expected = [
            ('a', 6),
            ('b', 7),
            ('c', 8),
            ('d', 9),
            ('e', 10),
            ('f', 11),
            ('g', 10),
            ('h', 9),
            ('i', 8),
            ('k', 7),
            ('l', 6),
        ];
        for (file, count) in expected {
            let n = all_coords()
                .filter(|(_, c)| c.to_notation().unwrap().starts_with(file))
                .count();
            assert_eq!(n, count, "file {}", file);
        }
    }

    #[test]
    fn test_notation_invalid_inputs() {
        assert!(HexCoord::from_notation("").is_none());
        assert!(HexCoord::from_notation("j1").is_none()); // j is skipped
        assert!(HexCoord::from_notation("a7").is_none()); // off-board
        assert!(HexCoord::from_notation("f12").is_none());
        assert!(HexCoord::from_notation("z3").is_none());
    }

    // -- Coordinate validation ------------------------------------------------

    #[test]
    fn test_center_is_valid() {
        assert!(HexCoord::new(0, 0).is_valid());
    }

    #[test]
    fn test_corners_valid() {
        // All 6 "corners" of the radius-5 hex
        let corners = [(5, 0), (-5, 0), (0, 5), (0, -5), (5, -5), (-5, 5)];
        for (q, r) in corners {
            assert!(
                HexCoord::new(q, r).is_valid(),
                "Corner ({q},{r}) should be valid"
            );
        }
    }

    #[test]
    fn test_off_board() {
        assert!(!HexCoord::new(6, 0).is_valid());
        assert!(!HexCoord::new(0, 6).is_valid());
        assert!(!HexCoord::new(3, 3).is_valid()); // q+r = 6 > 5
        assert!(!HexCoord::new(-3, -3).is_valid()); // q+r = -6 < -5
        assert!(!HexCoord::new(5, 5).is_valid());
    }

    #[test]
    fn test_total_valid_cells() {
        let mut count = 0;
        for q in -5..=5i8 {
            for r in -5..=5i8 {
                if HexCoord::new(q, r).is_valid() {
                    count += 1;
                }
            }
        }
        assert_eq!(count, 91);
    }

    // -- Index round-trips ----------------------------------------------------

    #[test]
    fn test_coord_to_index_roundtrip() {
        for idx in 0..NUM_CELLS {
            let coord = index_to_coord(idx);
            assert!(coord.is_valid());
            assert_eq!(coord_to_index(coord), Some(idx));
        }
    }

    #[test]
    fn test_all_valid_coords_have_index() {
        let mut seen = vec![false; NUM_CELLS];
        for q in -5..=5i8 {
            for r in -5..=5i8 {
                let c = HexCoord::new(q, r);
                if c.is_valid() {
                    let idx = coord_to_index(c).expect("valid coord should have index");
                    assert!(!seen[idx], "duplicate index {idx}");
                    seen[idx] = true;
                }
            }
        }
        assert!(seen.iter().all(|&b| b), "not all indices covered");
    }

    #[test]
    fn test_invalid_coord_no_index() {
        assert_eq!(coord_to_index(HexCoord::new(6, 0)), None);
        assert_eq!(coord_to_index(HexCoord::new(3, 3)), None);
    }

    // -- Directions -----------------------------------------------------------

    #[test]
    fn test_cardinal_directions_count() {
        assert_eq!(CARDINAL_DIRS.len(), 6);
    }

    #[test]
    fn test_diagonal_directions_count() {
        assert_eq!(DIAGONAL_DIRS.len(), 6);
    }

    #[test]
    fn test_center_has_six_cardinal_neighbors() {
        let center = HexCoord::new(0, 0);
        let count = CARDINAL_DIRS
            .iter()
            .filter(|&&(dq, dr)| center.step(dq, dr).is_some())
            .count();
        assert_eq!(count, 6);
    }

    #[test]
    fn test_center_has_six_diagonal_neighbors() {
        let center = HexCoord::new(0, 0);
        let count = DIAGONAL_DIRS
            .iter()
            .filter(|&&(dq, dr)| center.step(dq, dr).is_some())
            .count();
        assert_eq!(count, 6);
    }

    #[test]
    fn test_corner_has_fewer_neighbors() {
        let corner = HexCoord::new(5, 0);
        let card = CARDINAL_DIRS
            .iter()
            .filter(|&&(dq, dr)| corner.step(dq, dr).is_some())
            .count();
        assert!(
            card < 6,
            "corner should have fewer than 6 cardinal neighbors"
        );
    }

    // -- Initial position -----------------------------------------------------

    #[test]
    fn test_initial_piece_count() {
        let board = Board::new();
        let white = board.piece_count(Color::White);
        let black = board.piece_count(Color::Black);
        // Glinski: 9 pawns + 2 rooks + 2 knights + 3 bishops + 1 queen + 1 king = 18 per side
        assert_eq!(white, 18, "white should have 18 pieces");
        assert_eq!(black, 18, "black should have 18 pieces");
    }

    #[test]
    fn test_initial_kings() {
        let board = Board::new();
        assert_eq!(board.white_king, HexCoord::new(1, -5));
        assert_eq!(board.black_king, HexCoord::new(-1, 5));

        // Verify the king is actually there
        let wk = board.get(board.white_king);
        assert_eq!(wk, Some(Piece::new(PieceKind::King, Color::White)));
        let bk = board.get(board.black_king);
        assert_eq!(bk, Some(Piece::new(PieceKind::King, Color::Black)));
    }

    #[test]
    fn test_initial_white_to_move() {
        let board = Board::new();
        assert_eq!(board.side_to_move, Color::White);
    }

    #[test]
    fn test_initial_no_en_passant() {
        let board = Board::new();
        assert!(board.en_passant.is_none());
    }

    #[test]
    fn test_initial_white_pawns() {
        let board = Board::new();
        let expected: [(i8, i8); 9] = [
            (-4, -1),
            (-3, -1),
            (-2, -1),
            (-1, -1),
            (0, -1),
            (1, -2),
            (2, -3),
            (3, -4),
            (4, -5),
        ];
        for (q, r) in expected {
            let cell = board.get(HexCoord::new(q, r));
            assert_eq!(
                cell,
                Some(Piece::new(PieceKind::Pawn, Color::White)),
                "Expected white pawn at ({q},{r})"
            );
        }
    }

    #[test]
    fn test_initial_black_pawns() {
        let board = Board::new();
        let expected: [(i8, i8); 9] = [
            (4, 1),
            (3, 1),
            (2, 1),
            (1, 1),
            (0, 1),
            (-1, 2),
            (-2, 3),
            (-3, 4),
            (-4, 5),
        ];
        for (q, r) in expected {
            let cell = board.get(HexCoord::new(q, r));
            assert_eq!(
                cell,
                Some(Piece::new(PieceKind::Pawn, Color::Black)),
                "Expected black pawn at ({q},{r})"
            );
        }
    }

    #[test]
    fn test_initial_three_bishops_per_side() {
        let board = Board::new();
        let wb: Vec<_> = board
            .all_pieces(Color::White)
            .filter(|(_, p)| p.kind == PieceKind::Bishop)
            .collect();
        assert_eq!(wb.len(), 3, "White should have 3 bishops");
        let bb: Vec<_> = board
            .all_pieces(Color::Black)
            .filter(|(_, p)| p.kind == PieceKind::Bishop)
            .collect();
        assert_eq!(bb.len(), 3, "Black should have 3 bishops");
    }

    #[test]
    fn test_empty_board() {
        let board = Board::empty();
        for idx in 0..NUM_CELLS {
            assert!(board.cells[idx].is_none());
        }
        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(board.halfmove_clock, 0);
        assert_eq!(board.fullmove_number, 1);
    }

    // -- Symmetry: black is mirror of white -----------------------------------

    #[test]
    fn test_position_symmetry() {
        let board = Board::new();
        // For each white piece at (q, r), there should be the same kind of
        // black piece at (-q, -r).
        for (coord, piece) in board.all_pieces(Color::White) {
            let mirror = HexCoord::new(-coord.q, -coord.r);
            let mirrored = board.get(mirror);
            assert_eq!(
                mirrored,
                Some(Piece::new(piece.kind, Color::Black)),
                "White {} at {} should mirror to black {} at {}",
                piece.kind.symbol(),
                coord,
                piece.kind.symbol(),
                mirror
            );
        }
    }

    // -- Zobrist hashing ------------------------------------------------------

    #[test]
    fn test_zobrist_nonzero() {
        let board = Board::new();
        assert_ne!(
            board.zobrist_hash, 0,
            "starting position hash should not be zero"
        );
    }

    #[test]
    fn test_zobrist_changes_on_set() {
        let mut board = Board::new();
        let h1 = board.zobrist_hash;

        // Remove a pawn
        let pawn_coord = HexCoord::new(0, -1);
        board.set(pawn_coord, None);
        let h2 = board.zobrist_hash;
        assert_ne!(h1, h2, "hash should change when removing a piece");

        // Put it back
        board.set(pawn_coord, Some(Piece::new(PieceKind::Pawn, Color::White)));
        let h3 = board.zobrist_hash;
        assert_eq!(h1, h3, "hash should restore when piece is put back");
    }

    #[test]
    fn test_zobrist_deterministic() {
        let b1 = Board::new();
        let b2 = Board::new();
        assert_eq!(
            b1.zobrist_hash, b2.zobrist_hash,
            "same position should have same hash"
        );
    }

    #[test]
    fn test_zobrist_empty_vs_start() {
        let empty = Board::empty();
        let start = Board::new();
        assert_ne!(empty.zobrist_hash, start.zobrist_hash);
    }

    // -- get / set / all_pieces -----------------------------------------------

    #[test]
    fn test_set_and_get() {
        let mut board = Board::empty();
        let coord = HexCoord::new(2, 1);
        let piece = Piece::new(PieceKind::Rook, Color::Black);
        board.set(coord, Some(piece));
        assert_eq!(board.get(coord), Some(piece));
    }

    #[test]
    fn test_all_pieces_count() {
        let board = Board::new();
        let white_count = board.all_pieces(Color::White).count();
        let black_count = board.all_pieces(Color::Black).count();
        assert_eq!(white_count, 18);
        assert_eq!(black_count, 18);
    }

    #[test]
    fn test_king_pos() {
        let board = Board::new();
        assert_eq!(board.king_pos(Color::White), HexCoord::new(1, -5));
        assert_eq!(board.king_pos(Color::Black), HexCoord::new(-1, 5));
    }

    #[test]
    fn test_set_updates_king_cache() {
        let mut board = Board::empty();
        let coord = HexCoord::new(3, -1);
        board.set(coord, Some(Piece::new(PieceKind::King, Color::White)));
        assert_eq!(board.king_pos(Color::White), coord);
    }

    // -- Display --------------------------------------------------------------

    #[test]
    fn test_display_does_not_panic() {
        let board = Board::new();
        let s = format!("{}", board);
        assert!(!s.is_empty());
    }

    // -- Color ----------------------------------------------------------------

    #[test]
    fn test_opponent() {
        assert_eq!(Color::White.opponent(), Color::Black);
        assert_eq!(Color::Black.opponent(), Color::White);
    }

    // -- HexCoord step --------------------------------------------------------

    #[test]
    fn test_step_valid() {
        let c = HexCoord::new(0, 0);
        assert_eq!(c.step(1, 0), Some(HexCoord::new(1, 0)));
    }

    #[test]
    fn test_step_off_board() {
        let c = HexCoord::new(5, 0);
        assert_eq!(c.step(1, 0), None); // would go to (6, 0)
    }
}
