use agroai_core::similarity::{
    cosine_similarity_batch_value, cosine_similarity_value, l2_normalize_value, top_k_value,
};
use ndarray::array;

#[test]
fn cosine_similarity_identity_is_one() {
    let v = vec![0.5f32, 0.5, 0.5, 0.5];
    let sim = cosine_similarity_value(&v, &v).expect("cosine should succeed");
    assert!((sim - 1.0).abs() <= 1e-6);
}

#[test]
fn batch_matches_manual_dot_for_normalized_vectors() {
    let query = vec![1.0f32, 0.0, 0.0];
    let database = array![[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]];

    let scores = cosine_similarity_batch_value(&query, database.view()).expect("batch should work");
    assert_eq!(scores.len(), 3);
    assert!((scores[0] - 1.0).abs() <= 1e-6);
    assert!(scores[1].abs() <= 1e-6);
    assert!((scores[2] - 0.5).abs() <= 1e-6);
}

#[test]
fn top_k_returns_sorted_descending() {
    let scores = vec![0.1f32, 0.9, 0.3, 0.8];
    let (idx, vals) = top_k_value(&scores, 2);
    assert_eq!(idx, vec![1, 3]);
    assert_eq!(vals, vec![0.9, 0.8]);
}

#[test]
fn l2_normalize_returns_unit_norm() {
    let v = vec![3.0f32, 4.0];
    let n = l2_normalize_value(&v).expect("normalize should succeed");
    let norm = (n[0] * n[0] + n[1] * n[1]).sqrt();
    assert!((norm - 1.0).abs() <= 1e-6);
}
