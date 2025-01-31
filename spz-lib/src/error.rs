use std::{fmt, io};

#[derive(Debug)]
pub enum SpzError {
    ParseSplat(String),
    EmptyGaussianCloud,
    SerializePackedGaussians(String),
    ZstdCompress(String),
    ZstdDecompress(String),
    DeserializePackedGaussians(String),
    EmptyPackedGaussians,
    IoError(io::Error),
}

impl fmt::Display for SpzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpzError::ParseSplat(e) => {
                write!(f, "Failed to parse splats from the buffer: {}", e)
            }
            SpzError::EmptyGaussianCloud => {
                write!(f, "The Gaussian cloud is empty.")
            }
            SpzError::SerializePackedGaussians(e) => {
                write!(f, "Failed to serialize packed gaussians: {}", e)
            }
            SpzError::ZstdCompress(e) => {
                write!(f, "Zstandard compression failed: {}", e)
            }
            SpzError::ZstdDecompress(e) => {
                write!(f, "Zstandard decompression failed: {}", e)
            }
            SpzError::DeserializePackedGaussians(e) => {
                write!(f, "Failed to deserialize packed gaussians: {}", e)
            }
            SpzError::EmptyPackedGaussians => {
                write!(f, "There are no packed gaussians available.")
            }
            SpzError::IoError(e) => {
                write!(f, "An I/O error occurred: {}", e)
            }
        }
    }
}

impl std::error::Error for SpzError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SpzError::IoError(e) => Some(e),
            _ => None,
        }
    }
}
