//! Which Apple GPU generation a Mac has, from the architecture name MLX's
//! Metal device reports (`applegpu_g17s` on an M5 Pro).
//!
//! The name ends in a two-digit generation and a class letter: `g` for base
//! and Pro chips, `s` for Max, `d` for Ultra, `p` for phones. MLX reads it the
//! same way and turns on its M5 tensor-unit kernels from generation 17 (18 on
//! phones); this module mirrors that rule so Astronomical's choices follow
//! the kernels MLX actually runs.

/// The first Mac GPU generation with the M5's tensor units.
const M5_GPU_GENERATION: u32 = 17;
/// Phones reached the same units one generation later.
const PHONE_GPU_GENERATION_WITH_M5_UNITS: u32 = 18;

/// The generation in an architecture name, or `None` when the name does not
/// end in two digits and a class letter.
#[must_use]
pub fn apple_gpu_generation(architecture_name: &str) -> Option<u32> {
    let name_bytes = architecture_name.as_bytes();
    let [.., tens_digit, ones_digit, class_letter] = name_bytes else {
        return None;
    };
    if !tens_digit.is_ascii_digit() || !ones_digit.is_ascii_digit() || class_letter.is_ascii_digit()
    {
        return None;
    }
    Some(u32::from(tens_digit - b'0') * 10 + u32::from(ones_digit - b'0'))
}

/// Whether the GPU named is an M5 or newer: one with the tensor units MLX's
/// M5 kernels use. A name that cannot be read counts as older, so the
/// engine keeps the choices measured on every Mac.
#[must_use]
pub fn is_m5_or_newer_gpu(architecture_name: &str) -> bool {
    let Some(generation) = apple_gpu_generation(architecture_name) else {
        return false;
    };
    let first_generation_with_m5_units = if architecture_name.ends_with('p') {
        PHONE_GPU_GENERATION_WITH_M5_UNITS
    } else {
        M5_GPU_GENERATION
    };
    generation >= first_generation_with_m5_units
}

#[cfg(test)]
#[path = "apple_gpu_generation_tests.rs"]
mod apple_gpu_generation_tests;
