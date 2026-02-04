pub mod common;
pub mod error;
mod structures;

#[cfg(feature = "bytes")]
use bytes::Bytes;
use common::clamp_u8;
use common::degree_for_dim;
use common::dim_for_degree;
use common::half_to_float;
use common::inv_sigmoid;
use common::normalize_quat;
use common::plus_quat;
use common::quantize_sh;
use common::sigmoid;
use common::times_quat;
use common::unquantize_sh;
use error::SpzError;
use foldhash::HashMap;
use foldhash::HashMapExt;
use std::io::{Cursor, Write};
use structures::DpGaussians;
use structures::GaussianCloud;
use structures::PackedGaussians;
use structures::PackedGaussiansHeader;
use structures::FLAG_ANTIALIASED;
use structures::MAGIC;
use structures::VERSION;
use zerocopy::FromBytes;
use zerocopy::IntoBytes;
use zerocopy::F32;
use zerocopy::LE;
use zstd::stream::{decode_all, Encoder};

impl DpGaussians<'_> {
    fn uses_float16(&self) -> bool {
        let f16_len = self.num_points as usize * 3 * 2;
        self.positions.len() == f16_len
    }
}

fn pack_gaussians(gc: &GaussianCloud) -> PackedGaussians {
    let fractional_bits = 12;
    let color_scale = 0.15_f32;
    let sh_dim = dim_for_degree(gc.sh_degree);
    let sf = (1 << fractional_bits) as f32;
    let color_factor = color_scale * 255.0;
    let color_offset = 127.5; // 0.5 * 255.0

    let mut positions: Vec<u8> = Vec::with_capacity(gc.positions.len() * 3);
    positions.extend(gc.positions.iter().flat_map(|&val| {
        let fixed = (val * sf).round() as i32;
        [
            (fixed & 0xFF) as u8,
            ((fixed >> 8) & 0xFF) as u8,
            ((fixed >> 16) & 0xFF) as u8,
        ]
    }));

    let mut scales: Vec<u8> = Vec::with_capacity(gc.scales.len());
    scales.extend(gc.scales.iter().map(|&s| clamp_u8((s + 10.0) * 16.0)));

    let mut rotations: Vec<u8> = Vec::with_capacity((gc.rotations.len() / 4) * 3);
    rotations.extend(gc.rotations.chunks_exact(4).flat_map(|quat| {
        let (rx, ry, rz, rw) = (quat[0], quat[1], quat[2], quat[3]);
        let mut q = normalize_quat((rx, ry, rz, rw));
        let scale = if q.3 < 0.0 { -127.5 } else { 127.5 };
        q = times_quat(q, scale);
        q = plus_quat(q, (127.5, 127.5, 127.5, 127.5));
        [clamp_u8(q.0), clamp_u8(q.1), clamp_u8(q.2)]
    }));

    let mut alphas: Vec<u8> = Vec::with_capacity(gc.alphas.len());
    alphas.extend(gc.alphas.iter().map(|&a| clamp_u8(sigmoid(a) * 255.0)));

    let mut colors: Vec<u8> = Vec::with_capacity(gc.colors.len());
    colors.extend(
        gc.colors
            .iter()
            .map(|&c| clamp_u8(c * color_factor + color_offset)),
    );

    let sh = if gc.sh_degree > 0 {
        let sh_per_point = sh_dim * 3;
        let mut sh_vec: Vec<u8> = Vec::with_capacity(gc.sh.len());
        sh_vec.extend(gc.sh.chunks_exact(sh_per_point).flat_map(|chunk| {
            chunk.iter().enumerate().map(|(j, &x)| {
                let bucket = if j < 9 { 8 } else { 16 };
                quantize_sh(x, bucket)
            })
        }));
        sh_vec
    } else {
        Vec::new()
    };

    PackedGaussians {
        num_points: gc.num_points,
        sh_degree: gc.sh_degree,
        fractional_bits,
        antialiased: gc.antialiased,
        positions,
        scales,
        rotations,
        alphas,
        colors,
        sh,
    }
}

#[inline]
fn parse_3bytes(bytes: &[u8]) -> f32 {
    let b0 = bytes[0] as u32;
    let b1 = bytes[1] as u32;
    let b2 = bytes[2] as u32;
    let mut fixed = b0 | (b1 << 8) | (b2 << 16);
    if (fixed & 0x0080_0000) != 0 {
        fixed |= 0xFF00_0000; // Sign extend
    }
    fixed as i32 as f32
}

fn unpack_gaussians(pg: &DpGaussians) -> GaussianCloud {
    let np = pg.num_points as usize;
    let sh_dim = dim_for_degree(pg.sh_degree);
    let mut cloud = GaussianCloud {
        num_points: pg.num_points,
        sh_degree: pg.sh_degree,
        antialiased: pg.antialiased,
        positions: vec![0.0; np * 3],
        scales: vec![0.0; np * 3],
        rotations: vec![0.0; np * 4],
        alphas: vec![0.0; np],
        colors: vec![0.0; np * 3],
        sh: vec![0.0; np * sh_dim * 3],
    };

    let uses_f16 = pg.uses_float16();
    if uses_f16 {
        // Process positions as f16 chunks of 6 bytes per point
        for (i, chunk) in pg.positions.chunks_exact(6).enumerate() {
            let x = u16::from_be_bytes([chunk[0], chunk[1]]);
            let y = u16::from_be_bytes([chunk[2], chunk[3]]);
            let z = u16::from_be_bytes([chunk[4], chunk[5]]);

            cloud.positions[i * 3] = half_to_float(x);
            cloud.positions[i * 3 + 1] = half_to_float(y);
            cloud.positions[i * 3 + 2] = half_to_float(z);
        }
    } else {
        // Process positions
        let scale = 1.0 / ((1 << pg.fractional_bits) as f32);
        for (i, chunk) in pg.positions.chunks_exact(9).enumerate() {
            let x = parse_3bytes(&chunk[0..3]) * scale;
            let y = parse_3bytes(&chunk[3..6]) * scale;
            let z = parse_3bytes(&chunk[6..9]) * scale;

            cloud.positions[i * 3] = x;
            cloud.positions[i * 3 + 1] = y;
            cloud.positions[i * 3 + 2] = z;
        }
    }

    // Process scales
    cloud
        .scales
        .iter_mut()
        .zip(pg.scales.iter())
        .for_each(|(s, &pg_s)| {
            *s = pg_s as f32 / 16.0 - 10.0;
        });

    // Process rotations
    let rotation_scale = 1.0 / 127.5;
    pg.rotations
        .chunks_exact(3)
        .zip(cloud.rotations.chunks_exact_mut(4))
        .for_each(|(r_chunk, rot_chunk)| {
            let x = r_chunk[0] as f32 * rotation_scale - 1.0;
            let y = r_chunk[1] as f32 * rotation_scale - 1.0;
            let z = r_chunk[2] as f32 * rotation_scale - 1.0;

            let rr = 1.0 - (x * x + y * y + z * z);
            let w = if rr < 0.0 { 0.0 } else { rr.sqrt() };

            rot_chunk[0] = x;
            rot_chunk[1] = y;
            rot_chunk[2] = z;
            rot_chunk[3] = w;
        });

    // Process alphas
    cloud
        .alphas
        .iter_mut()
        .zip(pg.alphas.iter())
        .for_each(|(a, &pg_a)| {
            *a = inv_sigmoid(pg_a as f32 / 255.0);
        });

    // Process colors
    let color_scale = 0.15;
    cloud
        .colors
        .iter_mut()
        .zip(pg.colors.iter())
        .for_each(|(c, &pg_c)| {
            *c = (pg_c as f32 / 255.0 - 0.5) / color_scale;
        });

    // Process SH coefficients
    cloud
        .sh
        .iter_mut()
        .zip(pg.sh.iter())
        .for_each(|(sh, &pg_sh)| {
            *sh = unquantize_sh(pg_sh);
        });

    cloud
}

#[inline]
fn next_line<'b>(buffer: &'b [u8], offset: &mut usize) -> Option<&'b [u8]> {
    if *offset >= buffer.len() {
        return None;
    }
    let start = *offset;

    match memchr::memchr(b'\n', &buffer[*offset..]) {
        Some(pos) => {
            *offset = start + pos + 1;
            Some(&buffer[start..start + pos])
        }
        None => {
            *offset = buffer.len();
            Some(&buffer[start..])
        }
    }
}

#[inline(always)]
fn idx_of(hm: &HashMap<&str, usize>, name: &str) -> Result<usize, SpzError> {
    hm.get(name)
        .cloned()
        .ok_or_else(|| SpzError::ParseSplat(format!("Missing required field: {}", name)))
}

#[inline(always)]
fn bytes_to_f32(data: &[u8], field_name: &str) -> Result<f32, SpzError> {
    let (val, _rest) = <F32<LE>>::read_from_prefix(data)
        .map_err(|_| SpzError::ParseSplat(format!("Conversion error for {}", field_name)))?;
    Ok(val.get())
}

#[inline(never)]
fn parse_splat(raw_data: &[u8]) -> Result<GaussianCloud, SpzError> {
    let mut offset = 0;

    // Line #1: "ply"
    let line1 = next_line(raw_data, &mut offset)
        .ok_or_else(|| SpzError::ParseSplat("No 'ply' line".to_string()))?;
    if line1 != b"ply" {
        return Err(SpzError::ParseSplat(
            "Not a .ply file (missing 'ply' header)".to_string(),
        ));
    }

    // Line #2: "format binary_little_endian 1.0"
    let line2 = next_line(raw_data, &mut offset)
        .ok_or_else(|| SpzError::ParseSplat("Missing format line".to_string()))?;
    if line2 != b"format binary_little_endian 1.0" {
        return Err(SpzError::ParseSplat(
            "Unsupported .ply format (only binary_little_endian 1.0 is supported)".to_string(),
        ));
    }

    // Line #3: "element vertex N"
    let line3 = next_line(raw_data, &mut offset)
        .ok_or_else(|| SpzError::ParseSplat("Missing 'element vertex' line".to_string()))?;
    if !line3.starts_with(b"element vertex ") {
        return Err(SpzError::ParseSplat(
            "Missing 'element vertex' definition".to_string(),
        ));
    }
    let num_str = &line3[b"element vertex ".len()..];
    let num_points: usize = {
        let s = std::str::from_utf8(num_str)
            .map_err(|e| SpzError::ParseSplat(format!("UTF-8 error: {}", e)))?
            .trim();
        s.parse()
            .map_err(|e| SpzError::ParseSplat(format!("Parse error: {}", e)))?
    };
    if num_points == 0 {
        return Ok(GaussianCloud::default());
    }

    let mut field_map: HashMap<&str, usize> = HashMap::with_capacity(64);
    let mut field_count = 0;
    loop {
        let line = next_line(raw_data, &mut offset)
            .ok_or_else(|| SpzError::ParseSplat("No 'end_header' found before EOF".to_string()))?;

        // If line starts with "end_header", stop parsing the header
        if line.starts_with(b"end_header") {
            break;
        }

        // Only support "property float <name>"
        if !line.starts_with(b"property float ") {
            return Err(SpzError::ParseSplat(format!(
                "Unsupported property line: {:?}",
                line
            )));
        }
        let raw_name = &line[b"property float ".len()..];
        let name = std::str::from_utf8(raw_name)
            .map_err(|e| SpzError::ParseSplat(format!("UTF-8 error in field name: {}", e)))?;
        field_map.insert(name, field_count);
        field_count += 1;
    }
    let fields_per_vertex = field_count;

    // Retrieve field indices
    let ix = idx_of(&field_map, "x")?;
    let iy = idx_of(&field_map, "y")?;
    let iz = idx_of(&field_map, "z")?;
    let is0 = idx_of(&field_map, "scale_0")?;
    let is1 = idx_of(&field_map, "scale_1")?;
    let is2 = idx_of(&field_map, "scale_2")?;
    let ir0 = idx_of(&field_map, "rot_0")?;
    let ir1 = idx_of(&field_map, "rot_1")?;
    let ir2 = idx_of(&field_map, "rot_2")?;
    let ir3 = idx_of(&field_map, "rot_3")?;
    let iop = idx_of(&field_map, "opacity")?;
    let ic0 = idx_of(&field_map, "f_dc_0")?;
    let ic1 = idx_of(&field_map, "f_dc_1")?;
    let ic2 = idx_of(&field_map, "f_dc_2")?;

    // Optional spherical harmonics: f_rest_0 to f_rest_44 (up to 45)
    let mut sh_idx = Vec::with_capacity(45);
    for i in 0..45 {
        let nm = format!("f_rest_{}", i);
        if let Some(&found) = field_map.get(nm.as_str()) {
            sh_idx.push(found);
        } else {
            break;
        }
    }
    if sh_idx.len() % 3 != 0 {
        return Err(SpzError::ParseSplat(
            "Incomplete spherical harmonics fields".to_string(),
        ));
    }
    let sh_dim = sh_idx.len() / 3;

    // Calculate the expected length of the binary vertex data
    let expected_bytes = num_points
        .checked_mul(fields_per_vertex)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| SpzError::ParseSplat("Overflow in byte calculation".to_string()))?;

    // Check if there are enough bytes
    if raw_data.len() < offset + expected_bytes {
        return Err(SpzError::ParseSplat(format!(
            "Binary data is too short, need {} bytes, have {}",
            expected_bytes,
            raw_data.len() - offset
        )));
    }

    let data = &raw_data[offset..offset + expected_bytes];
    let mut cursor = 0;

    let mut cloud = GaussianCloud {
        num_points: num_points as i32,
        sh_degree: degree_for_dim(sh_dim),
        antialiased: false,
        positions: Vec::with_capacity(num_points * 3),
        scales: Vec::with_capacity(num_points * 3),
        rotations: Vec::with_capacity(num_points * 4),
        alphas: Vec::with_capacity(num_points),
        colors: Vec::with_capacity(num_points * 3),
        sh: Vec::with_capacity(num_points * sh_dim * 3),
    };

    // Precompute SH indices
    let sh_indices: Vec<_> = (0..sh_dim)
        .map(|j| (sh_idx[j], sh_idx[j + sh_dim], sh_idx[j + 2 * sh_dim]))
        .collect();
    // Process each vertex
    for _ in 0..num_points {
        // Ensure we have enough bytes left
        if cursor + fields_per_vertex * 4 > data.len() {
            return Err(SpzError::ParseSplat(
                "Unexpected end of binary data".to_string(),
            ));
        }

        // Extract the current vertex's data slice
        let vertex_data = &data[cursor..cursor + fields_per_vertex * 4];

        // Positions (using ix, iy, iz indices)
        let x = bytes_to_f32(&vertex_data[ix * 4..(ix + 1) * 4], "x")?;
        let y = bytes_to_f32(&vertex_data[iy * 4..(iy + 1) * 4], "y")?;
        let z = bytes_to_f32(&vertex_data[iz * 4..(iz + 1) * 4], "z")?;
        cloud.positions.extend_from_slice(&[x, y, z]);

        // Scales (using is0, is1, is2 indices)
        let s0 = bytes_to_f32(&vertex_data[is0 * 4..(is0 + 1) * 4], "scale_0")?;
        let s1 = bytes_to_f32(&vertex_data[is1 * 4..(is1 + 1) * 4], "scale_1")?;
        let s2 = bytes_to_f32(&vertex_data[is2 * 4..(is2 + 1) * 4], "scale_2")?;
        cloud.scales.extend_from_slice(&[s0, s1, s2]);

        // Rotations (using ir0, ir1, ir2, ir3 indices)
        let r0 = bytes_to_f32(&vertex_data[ir0 * 4..(ir0 + 1) * 4], "rot_0")?;
        let r1 = bytes_to_f32(&vertex_data[ir1 * 4..(ir1 + 1) * 4], "rot_1")?;
        let r2 = bytes_to_f32(&vertex_data[ir2 * 4..(ir2 + 1) * 4], "rot_2")?;
        let r3 = bytes_to_f32(&vertex_data[ir3 * 4..(ir3 + 1) * 4], "rot_3")?;
        cloud.rotations.extend_from_slice(&[r1, r2, r3, r0]);

        // Opacity (using iop index)
        let opacity = bytes_to_f32(&vertex_data[iop * 4..(iop + 1) * 4], "opacity")?;
        cloud.alphas.push(opacity);

        // Colors
        let c0 = bytes_to_f32(&vertex_data[ic0 * 4..(ic0 + 1) * 4], "f_dc_0")?;
        let c1 = bytes_to_f32(&vertex_data[ic1 * 4..(ic1 + 1) * 4], "f_dc_1")?;
        let c2 = bytes_to_f32(&vertex_data[ic2 * 4..(ic2 + 1) * 4], "f_dc_2")?;
        cloud.colors.extend_from_slice(&[c0, c1, c2]);

        // Spherical Harmonics
        for &(r_idx, g_idx, b_idx) in &sh_indices {
            let r = bytes_to_f32(&vertex_data[r_idx * 4..(r_idx + 1) * 4], "sh_r")?;
            let g = bytes_to_f32(&vertex_data[g_idx * 4..(g_idx + 1) * 4], "sh_g")?;
            let b = bytes_to_f32(&vertex_data[b_idx * 4..(b_idx + 1) * 4], "sh_b")?;
            cloud.sh.extend_from_slice(&[r, g, b]);
        }

        cursor += fields_per_vertex * 4;
    }

    Ok(cloud)
}

#[inline(never)]
fn compress_zstd(data: &[u8], level: u32, workers: u32) -> Result<Vec<u8>, SpzError> {
    let mut encoder = Encoder::new(Vec::new(), level as i32)
        .map_err(|e| SpzError::ZstdCompress(format!("Encoder creation failed: {}", e)))?;
    encoder
        .multithread(workers)
        .map_err(|e| SpzError::ZstdCompress(format!("Setting multithread failed: {}", e)))?;
    encoder
        .write_all(data)
        .map_err(|e| SpzError::ZstdCompress(format!("Writing data failed: {}", e)))?;

    let compressed_data = encoder
        .finish()
        .map_err(|e| SpzError::ZstdCompress(format!("Finalizing compression failed: {}", e)))?;
    Ok(compressed_data)
}

#[inline(never)]
fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>, SpzError> {
    decode_all(Cursor::new(data))
        .map_err(|e| SpzError::ZstdDecompress(format!("Decompression failed: {}", e)))
}

fn serialize_packed_gaussians(mut pg: PackedGaussians) -> Result<Vec<u8>, SpzError> {
    let header_size = std::mem::size_of::<PackedGaussiansHeader>();
    let data_size = pg.positions.len()
        + pg.alphas.len()
        + pg.colors.len()
        + pg.scales.len()
        + pg.rotations.len()
        + pg.sh.len();

    let mut out = Vec::with_capacity(header_size + data_size);

    let hdr = PackedGaussiansHeader {
        magic: MAGIC,
        version: VERSION,
        num_points: pg.num_points as u32,
        sh_degree: pg.sh_degree as u8,
        fractional_bits: pg.fractional_bits as u8,
        flags: if pg.antialiased { FLAG_ANTIALIASED } else { 0 },
        reserved: 0,
    };
    out.extend_from_slice(hdr.as_bytes());

    out.append(&mut pg.positions);
    out.append(&mut pg.alphas);
    out.append(&mut pg.colors);
    out.append(&mut pg.scales);
    out.append(&mut pg.rotations);
    out.append(&mut pg.sh);

    Ok(out)
}

fn deserialize_packed_gaussians(data: &[u8]) -> Result<DpGaussians<'_>, SpzError> {
    if data.len() < 16 {
        return Err(SpzError::DeserializePackedGaussians(
            "Corrupt header".to_string(),
        ));
    }
    let magic = u32::from_le_bytes(data[0..4].try_into().map_err(|e| {
        SpzError::DeserializePackedGaussians(format!("Magic number conversion error: {}", e))
    })?);
    let version = u32::from_le_bytes(data[4..8].try_into().map_err(|e| {
        SpzError::DeserializePackedGaussians(format!("Version conversion error: {}", e))
    })?);
    let num_points = u32::from_le_bytes(data[8..12].try_into().map_err(|e| {
        SpzError::DeserializePackedGaussians(format!("Number of points conversion error: {}", e))
    })?);
    let sh_degree = data[12];
    let fractional_bits = data[13];
    let flags = data[14];
    let _reserved = data[15];

    if magic != MAGIC {
        return Err(SpzError::DeserializePackedGaussians(
            "Invalid magic number".to_string(),
        ));
    }
    if version != VERSION {
        return Err(SpzError::DeserializePackedGaussians(
            "Unsupported version".to_string(),
        ));
    }

    let np = num_points as usize;
    let uses_f16 = false; // Only supporting version 2 (no float16)
    let antialiased = (flags & FLAG_ANTIALIASED) != 0;

    let pos_bytes = if uses_f16 { 6 } else { 9 };
    let positions_len = np * pos_bytes;
    let alphas_len = np;
    let colors_len = np * 3;
    let scales_len = np * 3;
    let rotations_len = np * 3;
    let dim = dim_for_degree(sh_degree as i32);
    let sh_len = np * dim * 3;

    let needed = 16 + positions_len + alphas_len + colors_len + scales_len + rotations_len + sh_len;
    if data.len() < needed {
        return Err(SpzError::DeserializePackedGaussians(format!(
            "Binary data is too short, need {} bytes, have {}",
            needed,
            data.len()
        )));
    }

    let mut offset = 16;
    let positions = &data[offset..offset + positions_len];
    offset += positions_len;
    let alphas = &data[offset..offset + alphas_len];
    offset += alphas_len;
    let colors = &data[offset..offset + colors_len];
    offset += colors_len;
    let scales = &data[offset..offset + scales_len];
    offset += scales_len;
    let rotations = &data[offset..offset + rotations_len];
    offset += rotations_len;
    let sh = &data[offset..offset + sh_len];

    Ok(DpGaussians {
        num_points: np as i32,
        sh_degree: sh_degree as i32,
        fractional_bits: fractional_bits as i32,
        antialiased,
        positions,
        scales,
        rotations,
        alphas,
        colors,
        sh,
    })
}

pub fn prepare_uncompressed(raw_data: &[u8]) -> Result<Vec<u8>, SpzError> {
    let gaussian_cloud = parse_splat(raw_data).map_err(|e| SpzError::ParseSplat(e.to_string()))?;
    if gaussian_cloud.num_points == 0 {
        return Err(SpzError::EmptyGaussianCloud);
    }
    let packed = pack_gaussians(&gaussian_cloud);
    let uncompressed = serialize_packed_gaussians(packed)
        .map_err(|e| SpzError::SerializePackedGaussians(e.to_string()))?;
    Ok(uncompressed)
}

fn write_ply(
    output: &mut Vec<u8>,
    cloud: &GaussianCloud,
    include_normals: bool,
) -> Result<(), SpzError> {
    let num_points = cloud.num_points as usize;
    let sh_dim = dim_for_degree(cloud.sh_degree);

    output.clear();
    output.extend_from_slice(b"ply\nformat binary_little_endian 1.0\n");
    writeln!(output, "element vertex {}", num_points).map_err(SpzError::IoError)?;
    output.extend_from_slice(b"property float x\nproperty float y\nproperty float z\n");
    if include_normals {
        output.extend_from_slice(b"property float nx\nproperty float ny\nproperty float nz\n");
    }
    output.extend_from_slice(
        b"property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n",
    );
    for i in 0..(sh_dim * 3) {
        writeln!(output, "property float f_rest_{}", i).map_err(SpzError::IoError)?;
    }
    output.extend_from_slice(
        b"property float opacity\n\
          property float scale_0\nproperty float scale_1\nproperty float scale_2\n\
          property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n\
          end_header\n",
    );

    // Estimate and reserve the required space
    let point_size = (3 + if include_normals { 3 } else { 0 } + 3 + (sh_dim * 3) + 1 + 3 + 4) * 4;
    output.reserve(num_points * point_size);

    let normals: [f32; 3] = [0.0, 0.0, 0.0];
    let mut sh_coeffs = Vec::with_capacity(3 * sh_dim);
    for i in 0..num_points {
        // Positions (x, y, z)
        let pos_slice: &[f32] = &cloud.positions[i * 3..i * 3 + 3];
        output.extend_from_slice(pos_slice.as_bytes());

        // Normals (nx, ny, nz) if included
        if include_normals {
            output.extend_from_slice(normals.as_bytes());
        }

        // Colors (f_dc_0, f_dc_1, f_dc_2)
        let color_slice: &[f32] = &cloud.colors[i * 3..i * 3 + 3];
        output.extend_from_slice(color_slice.as_bytes());

        // SH coefficients (f_rest_*)
        sh_coeffs.clear();
        for color_channel in 0..3 {
            for j in 0..sh_dim {
                let idx = (i * sh_dim + j) * 3 + color_channel;
                sh_coeffs.push(cloud.sh[idx]);
            }
        }
        output.extend_from_slice(sh_coeffs.as_bytes());

        // Opacity
        output.extend_from_slice(cloud.alphas[i].as_bytes());

        // Scales (scale_0, scale_1, scale_2)
        let scale_slice: &[f32] = &cloud.scales[i * 3..i * 3 + 3];
        output.extend_from_slice(scale_slice.as_bytes());

        // Rotations (w, x, y, z)
        let rot_slice = [
            cloud.rotations[i * 4 + 3],
            cloud.rotations[i * 4],
            cloud.rotations[i * 4 + 1],
            cloud.rotations[i * 4 + 2],
        ];
        output.extend_from_slice(rot_slice.as_bytes());
    }
    Ok(())
}

pub fn compress(
    raw_data: &[u8],
    compression_level: u32,
    workers: u32,
    output: &mut Vec<u8>,
) -> Result<(), SpzError> {
    let uncompressed = prepare_uncompressed(raw_data)?;
    let compressed = compress_zstd(&uncompressed, compression_level, workers)
        .map_err(|e| SpzError::ZstdCompress(e.to_string()))?;
    output.clear();
    output.extend_from_slice(&compressed);
    Ok(())
}

#[cfg(feature = "bytes")]
#[inline(never)]
pub fn compress_bytes(
    raw_data: Bytes,
    compression_level: u32,
    workers: u32,
    output: &mut Vec<u8>,
) -> Result<(), SpzError> {
    let uncompressed = prepare_uncompressed(&raw_data)?;
    let compressed = compress_zstd(&uncompressed, compression_level, workers)
        .map_err(|e| SpzError::ZstdCompress(e.to_string()))?;
    output.clear();
    output.extend_from_slice(&compressed);
    Ok(())
}

pub fn decompress(
    spz_data: &[u8],
    include_normals: bool,
    output: &mut Vec<u8>,
) -> Result<(), SpzError> {
    let uncompressed =
        decompress_zstd(spz_data).map_err(|e| SpzError::ZstdDecompress(e.to_string()))?;
    let packed = deserialize_packed_gaussians(&uncompressed)
        .map_err(|e| SpzError::DeserializePackedGaussians(e.to_string()))?;
    if packed.num_points == 0 {
        return Err(SpzError::EmptyPackedGaussians);
    }
    let cloud = unpack_gaussians(&packed);
    write_ply(output, &cloud, include_normals)
}

#[cfg(feature = "bytes")]
#[inline(never)]
pub fn decompress_bytes(
    spz_data: Bytes,
    include_normals: bool,
    output: &mut Vec<u8>,
) -> Result<(), SpzError> {
    let uncompressed =
        decompress_zstd(spz_data.as_ref()).map_err(|e| SpzError::ZstdDecompress(e.to_string()))?;
    let packed = deserialize_packed_gaussians(&uncompressed)
        .map_err(|e| SpzError::DeserializePackedGaussians(e.to_string()))?;
    if packed.num_points == 0 {
        return Err(SpzError::EmptyPackedGaussians);
    }
    let cloud = unpack_gaussians(&packed);
    write_ply(output, &cloud, include_normals)
}

cfg_if::cfg_if! {
if #[cfg(feature = "async")] {
    use tokio::task::spawn_blocking;
    use std::io;

    #[inline(never)]
    async fn compress_zstd_async(
        data: Bytes,
        level: u32,
        workers: u32,
    ) -> Result<Vec<u8>, SpzError> {
        spawn_blocking(move || compress_zstd(data.as_ref(), level, workers))
            .await
            .map_err(|e| SpzError::IoError(io::Error::other(format!("join error: {}", e))))?
    }

    #[inline(never)]
    async fn decompress_zstd_async(data: Bytes) -> Result<Vec<u8>, SpzError> {
        spawn_blocking(move || decompress_zstd(data.as_ref()))
            .await
            .map_err(|e| SpzError::IoError(io::Error::other(format!("join error: {}", e))))?
    }

    #[inline(never)]
    pub async fn compress_async(
        raw_data: Vec<u8>,
        compression_level: u32,
        workers: u32,
        output: &mut Vec<u8>,
    ) -> Result<(), SpzError> {
        // Offload CPU-bound prepare_uncompressed to blocking pool
        let uncompressed = spawn_blocking(move || prepare_uncompressed(&raw_data))
            .await
            .map_err(|e| SpzError::IoError(io::Error::other(format!("join error: {}", e))))??;
        let compressed =
            compress_zstd_async(Bytes::from(uncompressed), compression_level, workers).await?;
        output.clear();
        output.extend_from_slice(&compressed);
        Ok(())
    }

    #[cfg(feature = "bytes")]
    #[inline(never)]
    pub async fn compress_bytes_async(
        raw_data: Bytes,
        compression_level: u32,
        workers: u32,
        output: &mut Vec<u8>,
    ) -> Result<(), SpzError> {
        // Offload CPU-bound prepare_uncompressed to blocking pool
        let uncompressed = spawn_blocking(move || prepare_uncompressed(&raw_data))
            .await
            .map_err(|e| SpzError::IoError(io::Error::other(format!("join error: {}", e))))??;
        let compressed =
            compress_zstd_async(Bytes::from(uncompressed), compression_level, workers).await?;
        output.clear();
        output.extend_from_slice(&compressed);
        Ok(())
    }

    #[cfg(feature = "bytes")]
    #[inline(never)]
    pub async fn decompress_async(
        spz_data: Bytes,
        include_normals: bool,
        output: &mut Vec<u8>,
    ) -> Result<(), SpzError> {
        let uncompressed = decompress_zstd_async(spz_data).await?;
        let ply = spawn_blocking(move || {
            let packed = deserialize_packed_gaussians(&uncompressed)
                .map_err(|e| SpzError::DeserializePackedGaussians(e.to_string()))?;
            if packed.num_points == 0 {
                return Err(SpzError::EmptyPackedGaussians);
            }
            let cloud = unpack_gaussians(&packed);
            let mut out = Vec::new();
            write_ply(&mut out, &cloud, include_normals)?;
            Ok::<_, SpzError>(out)
        })
        .await
        .map_err(|e| SpzError::IoError(io::Error::other(format!("join error: {}", e))))??;

        output.clear();
        output.extend_from_slice(&ply);
        Ok(())
    }
}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, path::Path};

    #[test]
    fn test_missing_function_parse_splat_from_stream_is_here() {
        // minimal .ply, 0 vertices, no trailing newline, only "binary_little_endian" format
        let data =
            b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nend_header";
        let result = parse_splat(data);
        assert!(
            result.is_ok(),
            "It should parse an empty .ply OK (0 vertices)."
        );
        let cloud = result.unwrap();
        assert_eq!(cloud.num_points, 0);
    }

    fn create_test_ply() -> Vec<u8> {
        // minimal .ply with 1 vertex:
        let header = b"ply
format binary_little_endian 1.0
element vertex 1
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
    ";

        #[rustfmt::skip]
        let floats = [
            // x, y, z
            0.0f32, 0.1, 0.2,
            // scale_0, scale_1, scale_2
            0.01, 0.02, 0.03,
            // rot_0, rot_1, rot_2, rot_3
            0.0, 0.0, 0.0, 1.0,
            // opacity
            0.5,
            // f_dc_0, f_dc_1, f_dc_2
            0.2, 0.3, 0.4,
        ];

        let mut raw_ply = Vec::new();
        raw_ply.extend_from_slice(header);
        for &f in &floats {
            raw_ply.extend_from_slice(&f.to_le_bytes());
        }
        raw_ply
    }

    fn validate_output_ply(out_ply: &[u8]) {
        let text = String::from_utf8_lossy(out_ply);
        assert!(
            text.contains("element vertex 1"),
            "Output .ply missing 'element vertex 1'"
        );
    }

    #[test]
    fn test_compress_decompress() {
        let raw_ply = create_test_ply();

        let mut spz_data = Vec::new();
        compress(&raw_ply, 1, 1, &mut spz_data).expect("compress(...) failed");

        let mut out_ply = Vec::new();
        decompress(&spz_data, false, &mut out_ply).expect("decompress(...) failed");

        validate_output_ply(&out_ply);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_compress_decompress_async() {
        let raw_ply = create_test_ply();

        let spz_data = compress_zstd_async(Bytes::from(raw_ply), 1, 1)
            .await
            .expect("compress_zstd_async failed");

        let out_ply = decompress_zstd_async(Bytes::from(spz_data))
            .await
            .expect("decompress_zstd_async failed");

        validate_output_ply(&out_ply);
    }

    fn count_differences(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b).filter(|(x, y)| x != y).count()
            + a.len().saturating_sub(b.len())
            + b.len().saturating_sub(a.len())
    }

    fn find_first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
        a.iter().zip(b).position(|(x, y)| x != y)
    }

    fn format_bytes_hex_from(buffer: &[u8], start: usize) -> String {
        buffer
            .iter()
            .skip(start)
            .take(16)
            .map(|byte| format!("{:02x}", byte))
            .collect::<Vec<String>>()
            .join(" ")
    }

    #[test]
    fn test_compress_decompress_files() -> Result<(), SpzError> {
        let original_dir = Path::new("tests/original");
        // Should contain unpacked files from the C++ version (library must be compiled without march x86-64-v3)
        let decompressed_dir = Path::new("tests/decompressed");

        assert!(
            original_dir.is_dir(),
            "Original directory does not exist or is not a directory: {:?}",
            original_dir
        );
        assert!(
            decompressed_dir.is_dir(),
            "Decompressed directory does not exist or is not a directory: {:?}",
            decompressed_dir
        );

        for entry in fs::read_dir(original_dir).expect("Failed to read original directory") {
            let entry = entry.expect("Failed to read directory entry");
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("ply") {
                let filename = path.file_name().unwrap().to_string_lossy().into_owned();
                println!("Testing file: {}", filename);

                let original_ply = fs::read(&path).map_err(|_| {
                    SpzError::IoError(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to read original PLY file: {:?}", path),
                    ))
                })?;

                let mut compressed_output = Vec::new();
                compress(&original_ply, 3, 1, &mut compressed_output)?;

                let mut decompressed_output = Vec::new();
                decompress(&compressed_output, false, &mut decompressed_output)?;

                let expected_decompressed_path = decompressed_dir.join(&filename);
                let expected_decompressed =
                    fs::read(&expected_decompressed_path).map_err(|_| {
                        SpzError::IoError(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!(
                                "Failed to read expected decompressed PLY file: {:?}",
                                expected_decompressed_path
                            ),
                        ))
                    })?;

                // Compare the decompressed data to the expected PLY
                let decompressed_diff =
                    count_differences(&decompressed_output, &expected_decompressed);
                if decompressed_diff != 0 {
                    // Find first differing byte position
                    let first_diff_index =
                        find_first_difference(&decompressed_output, &expected_decompressed)
                            .unwrap_or_else(|| {
                                usize::min(decompressed_output.len(), expected_decompressed.len())
                            });

                    // Get 16-byte windows starting from first difference
                    let a_window = &decompressed_output
                        .iter()
                        .skip(first_diff_index)
                        .take(16)
                        .cloned()
                        .collect::<Vec<_>>();
                    let b_window = &expected_decompressed
                        .iter()
                        .skip(first_diff_index)
                        .take(16)
                        .cloned()
                        .collect::<Vec<_>>();

                    let window_diff_count = count_differences(a_window, b_window);

                    panic!(
                        "Decompressed data mismatch for file: {}.\n\
                     Total differing bytes: {}\n\
                     First difference at byte: {}\n\
                     Differing bytes in 16-byte window: {}\n\
                     Decompressed size: {}\n\
                     Expected size: {}\n\
                     Actual bytes starting at {}: {}\n\
                     Expected bytes starting at {}: {}",
                        filename,
                        decompressed_diff,
                        first_diff_index,
                        window_diff_count,
                        decompressed_output.len(),
                        expected_decompressed.len(),
                        first_diff_index,
                        format_bytes_hex_from(&decompressed_output, first_diff_index),
                        first_diff_index,
                        format_bytes_hex_from(&expected_decompressed, first_diff_index)
                    );
                }
            }
        }

        Ok(())
    }
}
