//! Beside-source unit contracts for the pure fused SDPA head-dimension
//! selector in `vision_attention.rs`. The MLX fused kernel specializes on a
//! fixed set of head widths; the mapping from a model's real head width to
//! the padded specialization is pure arithmetic over integers, so its
//! boundary behavior is pinned here without a graphics processor.

use super::vision_attention::fused_sdpa_head_dimension;

#[test]
fn should_keep_an_exact_kernel_specialization_unpadded() {
    // 64, 80, and 128 are the kernel's own widths; padding an exact match
    // would waste fused work for no benefit.
    assert_eq!(fused_sdpa_head_dimension(64), 64);
    assert_eq!(fused_sdpa_head_dimension(80), 80);
    assert_eq!(fused_sdpa_head_dimension(128), 128);
}

#[test]
fn should_pad_up_to_the_next_kernel_specialization() {
    // The pinned vision head width is 72: between 64 and 80, so it pads to
    // 80. Appended zeros are neutral for QK^T and for V output coordinates.
    assert_eq!(fused_sdpa_head_dimension(1), 64);
    assert_eq!(fused_sdpa_head_dimension(64 + 1), 80);
    assert_eq!(fused_sdpa_head_dimension(72), 80);
    assert_eq!(fused_sdpa_head_dimension(80 + 1), 128);
    assert_eq!(fused_sdpa_head_dimension(127), 128);
}

#[test]
fn should_run_unpadded_beyond_every_kernel_specialization() {
    // A width larger than the largest specialization cannot pad up; the
    // selector returns the width itself so the caller's slice-back stays a
    // no-op rather than underflowing.
    assert_eq!(fused_sdpa_head_dimension(129), 129);
    assert_eq!(fused_sdpa_head_dimension(256), 256);
}
