use bytes::Bytes;
use clap::{ArgGroup, Parser};
use spz_lib::common::ZSTD_MAX_COMPRESSION_LVL;
use spz_lib::{compress, compress_async, decompress, decompress_async};
use std::cmp::min;
use std::error::Error;
use std::fs;
use std::process;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "PLY Compressor/Decompressor",
    version = "1.0",
    author = "Denis Avvakumov",
    about = "Compresses or decompresses PLY files (splats)"
)]
#[command(group(
    ArgGroup::new("mode").required(true).args(&["compress", "decompress"])
))]
struct Cli {
    #[arg(short = 'e', long = "compress", help = "Enable compression mode.")]
    compress: bool,

    #[arg(short = 'd', long = "decompress", help = "Enable decompression mode.")]
    decompress: bool,

    #[arg(
        short = 'n',
        value_name = "INCLUDE_NORMALS",
        default_value = "false",
        long = "normals",
        help = "Include normals from the output PLY file (only valid with decompression)."
    )]
    include_normals: bool,

    #[arg(
        short = 'i',
        long = "input",
        value_name = "INPUT",
        required = true,
        help = "Path to the input file."
    )]
    input: String,

    #[arg(
        short = 'o',
        long = "output",
        value_name = "OUTPUT",
        required = true,
        help = "Path to the output file."
    )]
    output: String,

    #[arg(
        short = 'c',
        long = "compression-level",
        value_name = "LEVEL",
        default_value = "3",
        help = "Set the compression level."
    )]
    compression_level: u32,

    #[arg(
        short = 'w',
        long = "workers",
        value_name = "WORKERS",
        default_value = "3",
        help = "Set the workers count for ZSTD."
    )]
    workers: u32,

    #[arg(
        short = 'a',
        long = "async",
        default_value = "false",
        help = "Enable asynchronous compression/decompression mode."
    )]
    async_mode: bool,
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let cli = Cli::parse();

    if cli.include_normals && !cli.decompress {
        eprintln!("Error: --include-normals (-n) can only be used with decompression (-d).");
        process::exit(1);
    }

    let raw_data = fs::read(&cli.input).unwrap_or_else(|e| {
        eprintln!("Error reading input file {}: {}", cli.input, e);
        process::exit(1);
    });

    let mode = if cli.async_mode {
        "Asynchronous"
    } else {
        "Synchronous"
    };
    let op = if cli.compress {
        "Compression"
    } else {
        "Decompression"
    };

    // Print the header info.
    print!(
        "Mode: {} {}\nInput: {} | Output: {}",
        mode, op, cli.input, cli.output
    );
    if cli.compress {
        println!(" | Level: {}", cli.compression_level);
    } else {
        println!(
            " | {} normals from output",
            if cli.include_normals {
                "Including"
            } else {
                "Excluding"
            }
        );
    }

    let cmp_level = min(cli.compression_level, ZSTD_MAX_COMPRESSION_LVL);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let start = Instant::now();

    let result: Vec<u8> = if cli.async_mode {
        rt.block_on(async {
            let mut buf = Vec::new();
            if cli.compress {
                compress_async(raw_data, cmp_level, cli.workers, &mut buf)
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
            } else {
                decompress_async(Bytes::from(raw_data), cli.include_normals, &mut buf)
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
            }
            Ok::<Vec<u8>, Box<dyn Error + Send + Sync>>(buf)
        })?
    } else {
        let mut buf = Vec::new();
        if cli.compress {
            compress(&raw_data, cmp_level, cli.workers, &mut buf)
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
        } else {
            decompress(&raw_data, cli.include_normals, &mut buf)
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
        }
        buf
    };

    let elapsed = start.elapsed().as_millis();
    println!("{} Time: {} ms", op, elapsed);

    fs::write(&cli.output, &result).unwrap_or_else(|e| {
        eprintln!("Error writing output '{}': {}", cli.output, e);
        process::exit(1);
    });
    println!("Successfully wrote to '{}'.", cli.output);

    Ok(())
}
