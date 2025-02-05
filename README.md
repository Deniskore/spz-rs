## SPZ 404 Rust Edition
This is a low level Rust implementation of the [SPZ 404 library](https://github.com/404-Repo/spz)

## Build Targets and Requirements
The ZSTD package is required to build on all operating systems and architectures.
The mold linker is necessary for improved build performance on specific platforms as outlined below:

- **Windows (x86_64)**: Target CPU `x86-64-v3`.

- **Linux (x86_64)**: The target CPU is `x86-64-v3`, with the **mold linker** used for building.

- **Linux (aarch64)**: The target CPU features include NEON, SVE, AES, Dot Product, FP16, RCPC, and LSE extensions, with the **mold linker** used for building.

- **macOS (aarch64)**: Target CPU `target-cpu=apple-m1`.

## Features
- [x] Basic functionality
- [x] Async version

The async functions are disabled by default. Enable it by adding the "async" feature.

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
    raw_data: &[u8],
    compression_level: u32,
    workers: u32,
    output: &mut Vec<u8>,
) -> Result<(), SpzError>;

pub async fn decompress_async(
    spz_data: &[u8],
    include_normals: bool,
    output: &mut Vec<u8>,
) -> Result<(), SpzError>;
```
