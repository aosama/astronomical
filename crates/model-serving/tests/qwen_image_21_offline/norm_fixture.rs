// Fixed reference values for the Qwen-Image-2.1 zero-centered RMSNorm, derived from the diffusers
// QwenImage21ZeroCenterRMSNorm implementation. Do not edit by hand; they are cross-checked data.
//
// Pure oracle data consumed by the test module via `use`.
#![allow(dead_code)]
pub const NORM_DIM: usize = 8;
pub const NORM_EPS: f32 = 9.999999747e-06f32;
pub const NORM_INPUT: [f32; 8] = [
    1.000000000e+00f32,
    -2.000000000e+00f32,
    5.000000000e-01f32,
    3.000000000e+00f32,
    -2.500000000e-01f32,
    0.000000000e+00f32,
    1.500000000e+00f32,
    -1.000000000e+00f32,
];
pub const NORM_WEIGHT: [f32; 8] = [
    1.000000000e-01f32,
    -2.000000000e-01f32,
    0.000000000e+00f32,
    5.000000000e-01f32,
    -5.000000000e-01f32,
    2.500000000e-01f32,
    0.000000000e+00f32,
    -1.000000000e-01f32,
];
pub const ORACLE_NORM_OUT: [f32; 8] = [
    7.424095273e-01f32,
    -1.079868317e+00f32,
    3.374588788e-01f32,
    3.037129879e+00f32,
    -8.436471969e-02f32,
    0.000000000e+00f32,
    1.012376547e+00f32,
    -6.074259281e-01f32,
];
pub const ORACLE_NORM_ZERO_WEIGHT: [f32; 8] = [
    6.749177575e-01f32,
    -1.349835515e+00f32,
    3.374588788e-01f32,
    2.024753094e+00f32,
    -1.687294394e-01f32,
    0.000000000e+00f32,
    1.012376547e+00f32,
    -6.749177575e-01f32,
];
