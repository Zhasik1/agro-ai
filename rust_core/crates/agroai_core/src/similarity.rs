use crate::errors::CoreError;
use ndarray::ArrayView2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use wide::f32x8;

fn dot_simd(a: &[f32], b: &[f32]) -> Result<f32, CoreError> {
    if a.len() != b.len() {
        return Err(CoreError::Value(
            "Vectors must have the same length".to_string(),
        ));
    }

    let mut acc = f32x8::splat(0.0);
    let mut i = 0usize;
    while i + 8 <= a.len() {
        let lhs = f32x8::from([
            a[i],
            a[i + 1],
            a[i + 2],
            a[i + 3],
            a[i + 4],
            a[i + 5],
            a[i + 6],
            a[i + 7],
        ]);
        let rhs = f32x8::from([
            b[i],
            b[i + 1],
            b[i + 2],
            b[i + 3],
            b[i + 4],
            b[i + 5],
            b[i + 6],
            b[i + 7],
        ]);
        acc += lhs * rhs;
        i += 8;
    }

    let mut sum = acc.to_array().iter().sum::<f32>();
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    Ok(sum)
}

fn l2_norm_simd(v: &[f32]) -> f32 {
    let mut acc = f32x8::splat(0.0);
    let mut i = 0usize;
    while i + 8 <= v.len() {
        let lane = f32x8::from([
            v[i],
            v[i + 1],
            v[i + 2],
            v[i + 3],
            v[i + 4],
            v[i + 5],
            v[i + 6],
            v[i + 7],
        ]);
        acc += lane * lane;
        i += 8;
    }
    let mut sum = acc.to_array().iter().sum::<f32>();
    while i < v.len() {
        sum += v[i] * v[i];
        i += 1;
    }
    sum.sqrt()
}

fn to_vec1(arr: PyReadonlyArray1<'_, f32>) -> Vec<f32> {
    if let Ok(slice) = arr.as_slice() {
        slice.to_vec()
    } else {
        arr.as_array().iter().copied().collect()
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity_value(a: &[f32], b: &[f32]) -> Result<f32, CoreError> {
    let dot = dot_simd(a, b)?;
    let norm_a = l2_norm_simd(a);
    let norm_b = l2_norm_simd(b);
    let denom = norm_a * norm_b;

    if denom <= f32::EPSILON {
        return Err(CoreError::Value(
            "Cosine similarity is undefined for zero vectors".to_string(),
        ));
    }

    Ok(dot / denom)
}

/// Batch cosine-style score between a query vector and N database vectors.
///
/// If vectors are already L2-normalized, this equals cosine similarity.
pub fn cosine_similarity_batch_value(
    query: &[f32],
    database: ArrayView2<'_, f32>,
) -> Result<Vec<f32>, CoreError> {
    if database.ndim() != 2 {
        return Err(CoreError::Value("database must be 2D".to_string()));
    }
    let n = database.shape()[0];
    let d = database.shape()[1];
    if d != query.len() {
        return Err(CoreError::Value(
            "query length must match database second dimension".to_string(),
        ));
    }

    if let Some(slice) = database.as_slice() {
        let mut out = vec![0.0f32; n];
        out.par_iter_mut().enumerate().for_each(|(i, slot)| {
            let row = &slice[i * d..(i + 1) * d];
            *slot = dot_simd(row, query).unwrap_or(0.0);
        });
        Ok(out)
    } else {
        Ok(database
            .outer_iter()
            .into_par_iter()
            .map(|row| {
                let row_vec: Vec<f32> = row.iter().copied().collect();
                dot_simd(&row_vec, query).unwrap_or(0.0)
            })
            .collect())
    }
}

/// Top-k helper that returns sorted (indices, values) in descending order.
pub fn top_k_value(scores: &[f32], k: usize) -> (Vec<i64>, Vec<f32>) {
    if scores.is_empty() || k == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let take = k.min(indexed.len());
    indexed.truncate(take);

    let indices = indexed.iter().map(|(idx, _)| *idx as i64).collect();
    let values = indexed.iter().map(|(_, value)| *value).collect();
    (indices, values)
}

/// L2-normalize vector.
pub fn l2_normalize_value(v: &[f32]) -> Result<Vec<f32>, CoreError> {
    let norm = l2_norm_simd(v);
    if norm <= f32::EPSILON {
        return Err(CoreError::Value("Cannot normalize zero vector".to_string()));
    }
    Ok(v.iter().map(|x| *x / norm).collect())
}

#[pyfunction]
pub fn cosine_similarity(a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
    let a_vec = to_vec1(a);
    let b_vec = to_vec1(b);
    Ok(cosine_similarity_value(&a_vec, &b_vec)?)
}

#[pyfunction]
pub fn cosine_similarity_batch(
    query: PyReadonlyArray1<f32>,
    database: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray1<f32>>> {
    let py = query.py();
    let query_vec = to_vec1(query);
    let scores = cosine_similarity_batch_value(&query_vec, database.as_array())?;
    Ok(scores.into_pyarray_bound(py).unbind())
}

#[pyfunction]
pub fn top_k(scores: PyReadonlyArray1<f32>, k: usize) -> PyResult<(Py<PyArray1<i64>>, Py<PyArray1<f32>>)> {
    let py = scores.py();
    let scores_vec = to_vec1(scores);
    let (indices, values) = top_k_value(&scores_vec, k);
    Ok((
        indices.into_pyarray_bound(py).unbind(),
        values.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction]
pub fn l2_normalize(v: PyReadonlyArray1<f32>) -> PyResult<Py<PyArray1<f32>>> {
    let py = v.py();
    let vec = to_vec1(v);
    let normalized = l2_normalize_value(&vec)?;
    Ok(normalized.into_pyarray_bound(py).unbind())
}
