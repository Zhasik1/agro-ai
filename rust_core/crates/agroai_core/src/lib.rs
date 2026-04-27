use pyo3::prelude::*;

pub mod bbox;
pub mod errors;
pub mod hashing;
pub mod image_ops;
pub mod similarity;

#[pymodule]
fn agroai_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(image_ops::decode_image_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(image_ops::resize_bilinear, m)?)?;
    m.add_function(wrap_pyfunction!(image_ops::bgr_to_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(image_ops::normalize_imagenet, m)?)?;
    m.add_function(wrap_pyfunction!(image_ops::crop_bbox, m)?)?;

    m.add_function(wrap_pyfunction!(hashing::sha256_hex, m)?)?;
    m.add_function(wrap_pyfunction!(hashing::blake3_hex, m)?)?;
    m.add_function(wrap_pyfunction!(hashing::phash, m)?)?;

    m.add_function(wrap_pyfunction!(bbox::bbox_iou, m)?)?;
    m.add_function(wrap_pyfunction!(bbox::bbox_area, m)?)?;
    m.add_function(wrap_pyfunction!(bbox::nms, m)?)?;

    m.add_function(wrap_pyfunction!(similarity::cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::cosine_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::top_k, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::l2_normalize, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
