# Rust Core

`rust_core` hosts the high-performance native core for AgroAI. The Python app keeps full fallback behavior when this crate is not installed.

## Layout

- `crates/agroai_core`: PyO3 extension module exported as `agroai_core`.
- `crates/agroai_core/tests`: Rust integration tests.
- `crates/agroai_core/benches`: Criterion benchmarks.

## Build

```bash
cd rust_core/crates/agroai_core
maturin develop --release
```

## Test

```bash
cd rust_core
cargo test --all
```

## Benchmark

```bash
cd rust_core/crates/agroai_core
cargo bench
```

## Benchmark Notes

Run `cargo bench` locally to generate hardware-specific numbers.

| Benchmark | Rust result | Python baseline | Ratio |
| --- | --- | --- | --- |
| cosine batch | pending | pending | pending |
| NMS | pending | pending | pending |
| pHash | pending | pending | pending |
