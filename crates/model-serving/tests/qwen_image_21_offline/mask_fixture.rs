// Fixed reference values for the Qwen-Image-2.1 block-causal attention mask, derived from the
// diffusers block-causal mask implementation. Do not edit by hand; they are cross-checked data.
//
// Pure oracle data consumed by the test modules via `use`. Some constants are unused by any test
// and are intentionally allowed dead here rather than annotated individually.
#![allow(dead_code)]
pub const MASK_SEQ_LEN: usize = 16;
pub const EXPECTED_IMAGE_IDS: [i32; 16] = [
    -1i32, -1i32, 0i32, 0i32, 0i32, 0i32, -1i32, -1i32, -1i32, 1i32, 1i32, 1i32, 1i32, 1i32, 1i32,
    -1i32,
];

pub const ORACLE_MASK: [[u8; 16]; 16] = [
    [
        1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8,
    ],
];

pub const ORACLE_SEGMENTS: [(usize, usize, bool); 5] = [
    (0, 2, true),
    (2, 6, false),
    (6, 9, true),
    (9, 15, false),
    (15, 16, true),
];

pub const MASK_SEGMENT_COUNT: usize = 5;
pub const ORACLE_MASK_PADDED: [[u8; 16]; 16] = [
    [
        1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8,
    ],
    [
        1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8,
    ],
];

pub const PADDED_KEY_INDEX: usize = 15;
