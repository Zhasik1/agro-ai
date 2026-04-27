use agroai_core::bbox::nms_indices;
use agroai_core::hashing::phash_array;
use agroai_core::similarity::{cosine_similarity_batch_value, l2_normalize_value};
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array2, Array3};

fn make_db(n: usize, d: usize) -> (Vec<f32>, Array2<f32>) {
    let query_raw: Vec<f32> = (0..d).map(|i| ((i % 17) as f32 + 1.0) / 17.0).collect();
    let query = l2_normalize_value(&query_raw).unwrap_or_else(|_| query_raw.clone());

    let mut data = Vec::with_capacity(n * d);
    for row in 0..n {
        for col in 0..d {
            let v = (((row + col) % 23) as f32 + 1.0) / 23.0;
            data.push(v);
        }
    }

    let mut db = Array2::from_shape_vec((n, d), data).expect("shape is valid");
    for mut row in db.outer_iter_mut() {
        let normalized = l2_normalize_value(&row.to_vec()).unwrap_or_else(|_| row.to_vec());
        for (dst, src) in row.iter_mut().zip(normalized.iter()) {
            *dst = *src;
        }
    }

    (query, db)
}

fn bench_cosine_batch(c: &mut Criterion) {
    let (query, db) = make_db(10_000, 512);
    c.bench_function("cosine_similarity_batch_10k_x_512", |b| {
        b.iter(|| {
            let _ = cosine_similarity_batch_value(&query, db.view()).expect("batch should succeed");
        })
    });
}

fn bench_nms(c: &mut Criterion) {
    let n = 3000usize;
    let mut boxes = Array2::<f32>::zeros((n, 4));
    let mut scores = vec![0.0f32; n];
    for i in 0..n {
        let x = (i % 100) as f32;
        let y = ((i / 100) % 100) as f32;
        boxes[(i, 0)] = x;
        boxes[(i, 1)] = y;
        boxes[(i, 2)] = x + 20.0;
        boxes[(i, 3)] = y + 20.0;
        scores[i] = (n - i) as f32;
    }

    c.bench_function("nms_3k", |b| {
        b.iter(|| {
            let _ = nms_indices(boxes.view(), &scores, 0.5).expect("nms should succeed");
        })
    });
}

fn bench_phash(c: &mut Criterion) {
    let mut image = Array3::<u8>::zeros((256, 256, 3));
    for y in 0..256usize {
        for x in 0..256usize {
            image[(y, x, 0)] = (x % 255) as u8;
            image[(y, x, 1)] = (y % 255) as u8;
            image[(y, x, 2)] = ((x + y) % 255) as u8;
        }
    }

    c.bench_function("phash_256", |b| {
        b.iter(|| {
            let _ = phash_array(image.view()).expect("phash should succeed");
        })
    });
}

criterion_group!(benches, bench_cosine_batch, bench_nms, bench_phash);
criterion_main!(benches);
