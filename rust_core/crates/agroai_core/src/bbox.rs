use crate::errors::CoreError;
use ndarray::ArrayView2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, FromPyObject)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

fn width_height(b: BBox) -> (f32, f32) {
    ((b.x2 - b.x1).max(0.0), (b.y2 - b.y1).max(0.0))
}

/// Compute area of a bounding box.
pub fn bbox_area_value(b: BBox) -> f32 {
    let (w, h) = width_height(b);
    w * h
}

/// Compute IoU between two bounding boxes.
pub fn bbox_iou_value(a: BBox, b: BBox) -> f32 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter = inter_w * inter_h;

    let union = bbox_area_value(a) + bbox_area_value(b) - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn bbox_from_row(boxes: &ArrayView2<'_, f32>, idx: usize) -> BBox {
    BBox {
        x1: boxes[(idx, 0)],
        y1: boxes[(idx, 1)],
        x2: boxes[(idx, 2)],
        y2: boxes[(idx, 3)],
    }
}

/// Greedy NMS over boxes shaped (N, 4) and scores shaped (N,).
pub fn nms_indices(
    boxes: ArrayView2<'_, f32>,
    scores: &[f32],
    iou_threshold: f32,
) -> Result<Vec<i64>, CoreError> {
    if boxes.ndim() != 2 || boxes.shape()[1] != 4 {
        return Err(CoreError::Value("boxes must have shape (N, 4)".to_string()));
    }
    if !(0.0..=1.0).contains(&iou_threshold) {
        return Err(CoreError::Value(
            "iou_threshold must be in [0, 1]".to_string(),
        ));
    }

    let n = boxes.shape()[0];
    if n != scores.len() {
        return Err(CoreError::Value(
            "boxes and scores length mismatch".to_string(),
        ));
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap_or(Ordering::Equal));

    let mut keep: Vec<usize> = Vec::new();
    for &idx in &order {
        let candidate = bbox_from_row(&boxes, idx);

        let suppressed = if n > 1000 {
            keep.par_iter()
                .any(|&kept_idx| bbox_iou_value(candidate, bbox_from_row(&boxes, kept_idx)) >= iou_threshold)
        } else {
            keep.iter()
                .any(|&kept_idx| bbox_iou_value(candidate, bbox_from_row(&boxes, kept_idx)) >= iou_threshold)
        };

        if !suppressed {
            keep.push(idx);
        }
    }

    Ok(keep.into_iter().map(|idx| idx as i64).collect())
}

#[pyfunction]
pub fn bbox_iou(a: BBox, b: BBox) -> PyResult<f32> {
    Ok(bbox_iou_value(a, b))
}

#[pyfunction]
pub fn bbox_area(b: BBox) -> PyResult<f32> {
    Ok(bbox_area_value(b))
}

#[pyfunction]
pub fn nms(
    boxes: PyReadonlyArray2<f32>,
    scores: PyReadonlyArray1<f32>,
    iou_threshold: f32,
) -> PyResult<Py<PyArray1<i64>>> {
    let py = boxes.py();

    let score_vec = if let Ok(slice) = scores.as_slice() {
        slice.to_vec()
    } else {
        scores.as_array().iter().copied().collect()
    };

    let keep = nms_indices(boxes.as_array(), &score_vec, iou_threshold)?;
    Ok(keep.into_pyarray_bound(py).unbind())
}
