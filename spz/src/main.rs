use clap::{ArgGroup, Parser};
use spz_lib::{compress, decompress};
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::process;
use std::time::Instant;

use spz_lib::common::ZSTD_MAX_COMPRESSION_LVL;

#[derive(Parser, Debug)]
#[command(
    name = "PLY Compressor/Decompressor",
    version = "1.0",
    author = "Denis Avvakumov",
    about = "Compresses or decompresses PLY files (splats)"
)]
#[command(group(
    ArgGroup::new("mode")
        .required(true)
        .args(&["compress", "decompress"])
        .multiple(false)
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
        help = "Include normals from the output PLY file."
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
        short = 'с',
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
}

fn write_output(output_path: &str, data: &[u8]) {
    let mut output_file = match File::create(output_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error creating output file '{}': {}", output_path, e);
            process::exit(1);
        }
    };

    if let Err(e) = output_file.write_all(data) {
        eprintln!("Error writing to output file '{}': {}", output_path, e);
        process::exit(1);
    }

    println!("Successfully wrote to '{}'.", output_path);
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    if cli.include_normals && !cli.decompress {
        eprintln!("Error: --include-normals (-n) can only be used with decompression mode (-d).");
        std::process::exit(1);
    }

    let input_path = Path::new(&cli.input);
    let mut input_file = match File::open(input_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error opening input file {}: {}", cli.input, e);
            std::process::exit(1);
        }
    };

    let mut raw_data = Vec::new();
    if let Err(e) = input_file.read_to_end(&mut raw_data) {
        eprintln!("Error reading input file {}: {}", cli.input, e);
        std::process::exit(1);
    }

    if cli.compress {
        println!("Mode: Compression");
        println!("Input File: {}", cli.input);
        println!("Output File: {}", cli.output);
        println!("Compression Level: {}", cli.compression_level);

        let mut output = Vec::new();
        let start_time = Instant::now();
        compress(
            &raw_data,
            std::cmp::min(cli.compression_level, ZSTD_MAX_COMPRESSION_LVL),
            cli.workers,
            &mut output,
        )?;
        let duration = start_time.elapsed();
        println!("Compression Time: {} ms", duration.as_millis());

        write_output(&cli.output, &output);
    } else if cli.decompress {
        println!("Mode: Decompression");
        println!("Input File: {}", cli.input);
        println!("Output File: {}", cli.output);
        if cli.include_normals {
            println!("Excluding normals from the output PLY file.");
        }

        let mut output = Vec::new();
        let start_time = Instant::now();
        decompress(&raw_data, cli.include_normals, &mut output)?;
        let duration = start_time.elapsed();
        println!("Decompression Time: {} ms", duration.as_millis());

        write_output(&cli.output, &output);
    }

    Ok(())
}
