// Fixed reference values for the Qwen-Image-2.1 target-token mask and modulation row selection,
// derived from the diffusers token-metadata implementation. Do not edit by hand.
pub const MOD_SEQ_LEN: usize = 16;
pub const EXPECTED_TARGET_MASK: [bool; 16] = [
    false, false, false, false, false, false, false, false, false, true, true, true, true, true,
    true, false,
];

pub const ORACLE_ROW_MAP_B1: [[usize; 16]; 1] = [[
    1usize, 1usize, 1usize, 1usize, 1usize, 1usize, 1usize, 1usize, 1usize, 0usize, 0usize, 0usize,
    0usize, 0usize, 0usize, 1usize,
]];

pub const ORACLE_ROW_MAP_B2: [[usize; 16]; 2] = [
    [
        2usize, 2usize, 2usize, 2usize, 2usize, 2usize, 2usize, 2usize, 2usize, 0usize, 0usize,
        0usize, 0usize, 0usize, 0usize, 2usize,
    ],
    [
        2usize, 2usize, 2usize, 2usize, 2usize, 2usize, 2usize, 2usize, 2usize, 1usize, 1usize,
        1usize, 1usize, 1usize, 1usize, 2usize,
    ],
];

pub const ORACLE_ROW_MAP_NONE_B2: [[usize; 16]; 2] = [
    [
        0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize,
        0usize, 0usize, 0usize, 0usize, 0usize,
    ],
    [
        1usize, 1usize, 1usize, 1usize, 1usize, 1usize, 1usize, 1usize, 1usize, 1usize, 1usize,
        1usize, 1usize, 1usize, 1usize, 1usize,
    ],
];
