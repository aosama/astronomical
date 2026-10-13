// Fixed reference values for the Qwen-Image-2.1 KV-cache prefix slicing, derived from the diffusers
// forward slicing arithmetic. Do not edit by hand; they are cross-checked data.
//
// Pure oracle data consumed by the test module via `use`.
#![allow(dead_code)]
pub const CACHE_LAYOUT_COUNT: usize = 5;
pub const CACHE_SEQ_LEN: usize = 10;
pub const CACHE_MASKS: [[bool; 10]; 5] = [
    [false, false, true, true, true, true, true, true, true, true],
    [
        false, false, false, true, true, true, true, true, true, false,
    ],
    [true, true, true, true, true, true, true, true, true, true],
    [
        false, false, false, false, false, false, false, false, false, false,
    ],
    [false, true, true, true, true, true, true, true, true, true],
];
pub const CACHE_PREFIX_LENS: [usize; 5] = [2usize, 4usize, 0usize, 10usize, 1usize];
pub const CACHE_IS_VALID: [bool; 5] = [true, false, false, false, true];
