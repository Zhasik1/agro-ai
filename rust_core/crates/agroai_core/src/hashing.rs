use crate::errors::{ensure_rgb_shape, CoreError};
use ndarray::ArrayView3;
use numpy::PyReadonlyArray3;
use pyo3::prelude::*;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;

/// Compute SHA-256 hex digest for arbitrary bytes.
///
/// # Examples
///
/// ```rust
/// let digest = agroai_core::hashing::sha256_hex(b"abc").unwrap();
/// assert_eq!(digest, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
/// ```
#[pyfunction]
pub fn sha256_hex(bytes: &[u8]) -> PyResult<String> {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute BLAKE3 hex digest for arbitrary bytes.
#[pyfunction]
pub fn blake3_hex(bytes: &[u8]) -> PyResult<String> {
    Ok(blake3::hash(bytes).to_hex().to_string())
}

fn resize_bilinear_gray(input: &[f32], in_h: usize, in_w: usize, out_h: usize, out_w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_h * out_w];
    for oy in 0..out_h {
        let src_y = ((oy as f32 + 0.5) * (in_h as f32) / (out_h as f32) - 0.5).clamp(0.0, (in_h - 1) as f32);
        let y0 = src_y.floor() as usize;
        let y1 = (y0 + 1).min(in_h - 1);
        let wy = src_y - y0 as f32;

        for ox in 0..out_w {
            let src_x = ((ox as f32 + 0.5) * (in_w as f32) / (out_w as f32) - 0.5).clamp(0.0, (in_w - 1) as f32);
            let x0 = src_x.floor() as usize;
            let x1 = (x0 + 1).min(in_w - 1);
            let wx = src_x - x0 as f32;

            let p00 = input[y0 * in_w + x0];
            let p01 = input[y0 * in_w + x1];
            let p10 = input[y1 * in_w + x0];
            let p11 = input[y1 * in_w + x1];

            let top = p00 * (1.0 - wx) + p01 * wx;
            let bottom = p10 * (1.0 - wx) + p11 * wx;
            out[oy * out_w + ox] = top * (1.0 - wy) + bottom * wy;
        }
    }
    out
}

fn dct_top_left_8x8(input: &[f32], n: usize) -> [[f32; 8]; 8] {
    let mut out = [[0.0f32; 8]; 8];
    let pi = std::f32::consts::PI;

    let mut cos_x = [[0.0f32; 32]; 8];
    let mut cos_y = [[0.0f32; 32]; 8];
    for u in 0..8 {
        for x in 0..n {
            cos_x[u][x] = ((2.0 * x as f32 + 1.0) * u as f32 * pi / (2.0 * n as f32)).cos();
            cos_y[u][x] = cos_x[u][x];
        }
    }

    for u in 0..8 {
        for v in 0..8 {
            let mut sum = 0.0f32;
            for y in 0..n {
                for x in 0..n {
                    sum += input[y * n + x] * cos_x[u][x] * cos_y[v][y];
                }
            }

            let alpha_u = if u == 0 {
                (1.0 / n as f32).sqrt()
            } else {
                (2.0 / n as f32).sqrt()
            };
            let alpha_v = if v == 0 {
                (1.0 / n as f32).sqrt()
            } else {
                (2.0 / n as f32).sqrt()
            };
            out[u][v] = alpha_u * alpha_v * sum;
        }
    }

    out
}

/// Compute pHash for an RGB image. Returns a 16-char lowercase hex string.
pub fn phash_array(input: ArrayView3<'_, u8>) -> Result<String, CoreError> {
    ensure_rgb_shape(input.shape())?;

    let in_h = input.shape()[0];
    let in_w = input.shape()[1];
    let mut gray = vec![0.0f32; in_h * in_w];

    for y in 0..in_h {
        for x in 0..in_w {
            let r = input[(y, x, 0)] as f32;
            let g = input[(y, x, 1)] as f32;
            let b = input[(y, x, 2)] as f32;
            gray[y * in_w + x] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
    }

    let resized = resize_bilinear_gray(&gray, in_h, in_w, 32, 32);
    let dct = dct_top_left_8x8(&resized, 32);

    let mut coeffs = Vec::with_capacity(63);
    for u in 0..8 {
        for v in 0..8 {
            if u == 0 && v == 0 {
                continue;
            }
            coeffs.push(dct[u][v]);
        }
    }

    coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median = coeffs[coeffs.len() / 2];

    let mut bits = 0u64;
    let mut bit = 0u32;
    for u in 0..8 {
        for v in 0..8 {
            if dct[u][v] > median {
                bits |= 1u64 << bit;
            }
            bit += 1;
        }
    }

    Ok(format!("{bits:016x}"))
}

#[pyfunction]
pub fn phash(input: PyReadonlyArray3<u8>) -> PyResult<String> {
    Ok(phash_array(input.as_array())?)
}
