use crate::errors::{ensure_rgb_shape, CoreError};
use ndarray::{Array3, ArrayView3};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use wide::f32x8;

/// Decode image bytes (PNG/JPEG/WebP) into an RGB array (H, W, 3).
///
/// # Examples
///
/// ```no_run
/// use pyo3::Python;
///
/// Python::with_gil(|py| {
///     let png_bytes: Vec<u8> = std::fs::read("example.png").unwrap_or_default();
///     let _ = agroai_core::image_ops::decode_image_rgb(py, &png_bytes);
/// });
/// ```
pub fn decode_image_rgb_array(bytes: &[u8]) -> Result<Array3<u8>, CoreError> {
    let image = image::load_from_memory(bytes)
        .map_err(|err| CoreError::Value(format!("Failed to decode image bytes: {err}")))?;
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    Array3::from_shape_vec((height as usize, width as usize, 3), rgb.into_raw())
        .map_err(|err| CoreError::Runtime(format!("Failed to materialize RGB array: {err}")))
}

/// Resize an RGB uint8 image with bilinear interpolation.
pub fn resize_bilinear_rgb(
    input: ArrayView3<'_, u8>,
    target_h: usize,
    target_w: usize,
) -> Result<Array3<u8>, CoreError> {
    ensure_rgb_shape(input.shape())?;
    if target_h == 0 || target_w == 0 {
        return Err(CoreError::Value(
            "target_h and target_w must be greater than 0".to_string(),
        ));
    }

    let src_h = input.shape()[0];
    let src_w = input.shape()[1];
    let row_len = target_w * 3;
    let mut out = vec![0u8; target_h * row_len];

    out.par_chunks_mut(row_len).enumerate().for_each(|(dy, row)| {
        let src_y = ((dy as f32 + 0.5) * (src_h as f32) / (target_h as f32) - 0.5)
            .clamp(0.0, (src_h - 1) as f32);
        let y0 = src_y.floor() as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = src_y - y0 as f32;

        for dx in 0..target_w {
            let src_x = ((dx as f32 + 0.5) * (src_w as f32) / (target_w as f32) - 0.5)
                .clamp(0.0, (src_w - 1) as f32);
            let x0 = src_x.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = src_x - x0 as f32;

            for c in 0..3 {
                let p00 = input[(y0, x0, c)] as f32;
                let p01 = input[(y0, x1, c)] as f32;
                let p10 = input[(y1, x0, c)] as f32;
                let p11 = input[(y1, x1, c)] as f32;

                let top = p00 * (1.0 - wx) + p01 * wx;
                let bottom = p10 * (1.0 - wx) + p11 * wx;
                let value = top * (1.0 - wy) + bottom * wy;
                row[dx * 3 + c] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    });

    Array3::from_shape_vec((target_h, target_w, 3), out)
        .map_err(|err| CoreError::Runtime(format!("Failed to create resized array: {err}")))
}

/// Convert BGR uint8 image to RGB uint8 image.
pub fn bgr_to_rgb_array(input: ArrayView3<'_, u8>) -> Result<Array3<u8>, CoreError> {
    ensure_rgb_shape(input.shape())?;

    let h = input.shape()[0];
    let w = input.shape()[1];
    let mut out = Array3::<u8>::zeros((h, w, 3));

    for y in 0..h {
        for x in 0..w {
            out[(y, x, 0)] = input[(y, x, 2)];
            out[(y, x, 1)] = input[(y, x, 1)];
            out[(y, x, 2)] = input[(y, x, 0)];
        }
    }

    Ok(out)
}

/// Normalize an RGB uint8 image into ImageNet-normalized float32 CHW tensor.
pub fn normalize_imagenet_array(input: ArrayView3<'_, u8>) -> Result<Array3<f32>, CoreError> {
    ensure_rgb_shape(input.shape())?;

    let h = input.shape()[0];
    let w = input.shape()[1];
    let mut out = Array3::<f32>::zeros((3, h, w));
    let means = [0.485f32, 0.456f32, 0.406f32];
    let stds = [0.229f32, 0.224f32, 0.225f32];

    let inv_255 = f32x8::splat(1.0 / 255.0);

    for c in 0..3 {
        let mean_v = f32x8::splat(means[c]);
        let std_v = f32x8::splat(stds[c]);
        for y in 0..h {
            let mut x = 0usize;
            while x + 8 <= w {
                let lane = f32x8::from([
                    input[(y, x, c)] as f32,
                    input[(y, x + 1, c)] as f32,
                    input[(y, x + 2, c)] as f32,
                    input[(y, x + 3, c)] as f32,
                    input[(y, x + 4, c)] as f32,
                    input[(y, x + 5, c)] as f32,
                    input[(y, x + 6, c)] as f32,
                    input[(y, x + 7, c)] as f32,
                ]);
                let normalized = (lane * inv_255 - mean_v) / std_v;
                let vals = normalized.to_array();
                for i in 0..8 {
                    out[(c, y, x + i)] = vals[i];
                }
                x += 8;
            }

            while x < w {
                let value = input[(y, x, c)] as f32 / 255.0;
                out[(c, y, x)] = (value - means[c]) / stds[c];
                x += 1;
            }
        }
    }

    Ok(out)
}

/// Crop a bounding box from an RGB image. Coordinates are clipped to image bounds.
pub fn crop_bbox_array(
    input: ArrayView3<'_, u8>,
    bbox: (i32, i32, i32, i32),
) -> Result<Array3<u8>, CoreError> {
    ensure_rgb_shape(input.shape())?;

    let h = input.shape()[0] as i32;
    let w = input.shape()[1] as i32;

    let x1 = bbox.0.clamp(0, w);
    let y1 = bbox.1.clamp(0, h);
    let x2 = bbox.2.clamp(0, w);
    let y2 = bbox.3.clamp(0, h);

    if x2 <= x1 || y2 <= y1 {
        return Err(CoreError::Value(
            "Bounding box is empty after clipping".to_string(),
        ));
    }

    let out_h = (y2 - y1) as usize;
    let out_w = (x2 - x1) as usize;
    let mut out = Array3::<u8>::zeros((out_h, out_w, 3));

    for y in 0..out_h {
        for x in 0..out_w {
            for c in 0..3 {
                out[(y, x, c)] = input[((y1 as usize) + y, (x1 as usize) + x, c)];
            }
        }
    }

    Ok(out)
}

#[pyfunction]
pub fn decode_image_rgb(py: Python<'_>, bytes: &[u8]) -> PyResult<Py<PyArray3<u8>>> {
    let arr = decode_image_rgb_array(bytes)?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

#[pyfunction]
pub fn resize_bilinear(
    input: PyReadonlyArray3<u8>,
    target_h: u32,
    target_w: u32,
) -> PyResult<Py<PyArray3<u8>>> {
    let py = input.py();
    let arr = resize_bilinear_rgb(input.as_array(), target_h as usize, target_w as usize)?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

#[pyfunction]
pub fn bgr_to_rgb(input: PyReadonlyArray3<u8>) -> PyResult<Py<PyArray3<u8>>> {
    let py = input.py();
    let arr = bgr_to_rgb_array(input.as_array())?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

#[pyfunction]
pub fn normalize_imagenet(input: PyReadonlyArray3<u8>) -> PyResult<Py<PyArray3<f32>>> {
    let py = input.py();
    let arr = normalize_imagenet_array(input.as_array())?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

#[pyfunction]
pub fn crop_bbox(
    input: PyReadonlyArray3<u8>,
    bbox: (i32, i32, i32, i32),
) -> PyResult<Py<PyArray3<u8>>> {
    let py = input.py();
    let arr = crop_bbox_array(input.as_array(), bbox)?;
    Ok(arr.into_pyarray_bound(py).unbind())
}
