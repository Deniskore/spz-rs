## SPZ 404 Rust Edition
This is a low level Rust implementation of the [SPZ 404 library](https://github.com/404-Repo/spz)

## Build Targets and Requirements
The ZSTD package is required to build on all operating systems and architectures.

- **Windows (x86_64)**: Target CPU `x86-64-v3`.

- **Linux (x86_64)**: The target CPU is `x86-64-v3`.

- **Linux (aarch64)**: The target CPU features include NEON, SVE, AES, Dot Product, FP16, RCPC, and LSE extensions.

- **macOS (aarch64)**: Target CPU `target-cpu=apple-m1`.

## Features
- [x] Basic functionality
- [x] Async version
- [x] Bytes crate support

Features are disabled by default. Enable with Cargo features:
- Async: `--features async`
- Bytes API (sync only): `--features bytes`
- Async + Bytes: `--features async,bytes`

## Interface

```Rust
pub fn compress(
    raw_data: &[u8],
    compression_level: u32,
    workers: u32,
    output: &mut Vec<u8>,
) -> Result<(), SpzError>;

pub fn decompress(
    spz_input: &[u8],
    include_normals: bool,
    output: &mut Vec<u8>,
) -> Result<(), SpzError>;

pub async fn compress_async(
    raw_data: Vec<u8>,
    compression_level: u32,
    workers: u32,
    output: &mut Vec<u8>,
) -> Result<(), SpzError>;

pub async fn decompress_async(
    spz_data: &[u8],
    include_normals: bool,
    output: &mut Vec<u8>,
) -> Result<(), SpzError>;

// With `bytes` feature (sync):
pub fn compress_bytes(
    raw_data: bytes::Bytes,
    compression_level: u32,
    workers: u32,
    output: &mut Vec<u8>,
) -> Result<(), SpzError>;

pub fn decompress_bytes(
    spz_data: bytes::Bytes,
    include_normals: bool,
    output: &mut Vec<u8>,
) -> Result<(), SpzError>;

// With `async` + `bytes` features:
pub async fn compress_bytes_async(
    raw_data: bytes::Bytes,
    compression_level: u32,
    workers: u32,
    output: &mut Vec<u8>,
) -> Result<(), SpzError>;

pub async fn decompress_bytes_async(
    spz_data: bytes::Bytes,
    include_normals: bool,
    output: &mut Vec<u8>,
) -> Result<(), SpzError>;
```
