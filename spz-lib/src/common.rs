use half::f16;

pub const ZSTD_MAX_COMPRESSION_LVL: u32 = 22;

#[inline]
pub(crate) fn clamp_u8(x: f32) -> u8 {
    x.round().clamp(0.0, 255.0) as u8
}

#[inline]
pub(crate) fn quantize_sh(x: f32, bucket_size: i32) -> u8 {
    let q = (x * 128.0).round() as i32 + 128;
    let q = ((q + bucket_size / 2) / bucket_size) * bucket_size;
    q.clamp(0, 255) as u8
}

#[inline]
pub(crate) fn unquantize_sh(x: u8) -> f32 {
    (x as f32 - 128.0) / 128.0
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub(crate) fn inv_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

#[inline]
pub(crate) fn dim_for_degree(deg: i32) -> usize {
    match deg {
        0 => 0,
        1 => 3,
        2 => 8,
        3 => 15,
        _ => 0,
    }
}

#[inline]
pub(crate) const fn degree_for_dim(dim: usize) -> i32 {
    if dim < 3 {
        0
    } else if dim < 8 {
        1
    } else if dim < 15 {
        2
    } else {
        3
    }
}

#[inline]
pub(crate) fn half_to_float(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

#[inline]
pub(crate) fn normalize_quat(q: (f32, f32, f32, f32)) -> (f32, f32, f32, f32) {
    let norm = (q.0 * q.0 + q.1 * q.1 + q.2 * q.2 + q.3 * q.3).sqrt();
    (q.0 / norm, q.1 / norm, q.2 / norm, q.3 / norm)
}

#[inline]
pub(crate) const fn times_quat(a: (f32, f32, f32, f32), s: f32) -> (f32, f32, f32, f32) {
    ((a.0 * s), (a.1 * s), (a.2 * s), (a.3 * s))
}

#[inline]
pub(crate) const fn plus_quat(
    a: (f32, f32, f32, f32),
    b: (f32, f32, f32, f32),
) -> (f32, f32, f32, f32) {
    ((a.0 + b.0), (a.1 + b.1), (a.2 + b.2), (a.3 + b.3))
}
