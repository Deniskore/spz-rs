## SPZ 404 Rust Edition
This is a low level Rust implementation of the [SPZ 404 library](https://github.com/404-Repo/spz)

## Build Targets and Requirements

- **Windows (x86_64)**: Target CPU `x86-64-v3`.

- **Linux (x86_64)**: The target CPU is x86-64-v3, with the mold linker used for building.

- **Linux (aarch64)**: The target CPU features include NEON, SVE, Crypto, Dot Product, FP16, RCPC, and LSE extensions, with the mold linker used for building.

- **macOS (aarch64)**: Target CPU `target-cpu=apple-m1`.

## Features
- [x] Basic functionality
- [ ] Async version

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
```
