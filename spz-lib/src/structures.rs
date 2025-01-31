pub const MAGIC: u32 = 0x5053474E; // 'NGSP'
pub const VERSION: u32 = 2;
pub const FLAG_ANTIALIASED: u8 = 0x1;

#[derive(Debug, Default, Clone)]
pub struct GaussianCloud {
    pub num_points: i32,
    pub sh_degree: i32,
    pub antialiased: bool,
    pub positions: Vec<f32>,
    pub scales: Vec<f32>,
    pub rotations: Vec<f32>,
    pub alphas: Vec<f32>,
    pub colors: Vec<f32>,
    pub sh: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct PackedGaussians {
    pub num_points: i32,
    pub sh_degree: i32,
    pub fractional_bits: i32,
    pub antialiased: bool,
    pub positions: Vec<u8>,
    pub scales: Vec<u8>,
    pub rotations: Vec<u8>,
    pub alphas: Vec<u8>,
    pub colors: Vec<u8>,
    pub sh: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct DpGaussians<'a> {
    pub num_points: i32,
    pub sh_degree: i32,
    pub fractional_bits: i32,
    pub antialiased: bool,
    pub positions: &'a [u8],
    pub scales: &'a [u8],
    pub rotations: &'a [u8],
    pub alphas: &'a [u8],
    pub colors: &'a [u8],
    pub sh: &'a [u8],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PackedGaussiansHeader {
    pub magic: u32,
    pub version: u32,
    pub num_points: u32,
    pub sh_degree: u8,
    pub fractional_bits: u8,
    pub flags: u8,
    pub reserved: u8,
}
