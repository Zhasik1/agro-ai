use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("{0}")]
    Value(String),
    #[error("{0}")]
    Runtime(String),
}

impl From<CoreError> for PyErr {
    fn from(value: CoreError) -> Self {
        match value {
            CoreError::Value(message) => PyValueError::new_err(message),
            CoreError::Runtime(message) => PyRuntimeError::new_err(message),
        }
    }
}

pub fn ensure_rgb_shape(shape: &[usize]) -> Result<(), CoreError> {
    if shape.len() != 3 {
        return Err(CoreError::Value("Expected HxWx3 image tensor".to_string()));
    }
    if shape[2] != 3 {
        return Err(CoreError::Value("Expected 3 channels (RGB/BGR)".to_string()));
    }
    if shape[0] == 0 || shape[1] == 0 {
        return Err(CoreError::Value("Image dimensions must be non-zero".to_string()));
    }
    Ok(())
}
