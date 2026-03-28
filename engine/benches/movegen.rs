use criterion::{criterion_group, criterion_main, Criterion};
use hexchess_engine::board::Board;
use hexchess_engine::movegen::generate_legal_moves;

fn bench_legal_moves_starting(c: &mut Criterion) {
    let board = Board::new();
    c.bench_function("legal_moves_starting_position", |b| {
        b.iter(|| generate_legal_moves(&board))
    });
}

criterion_group!(benches, bench_legal_moves_starting);
criterion_main!(benches);
