//! How deep each contraction part of a projection may be, from the model's own shape.
//!
//! The Neural Engine runs a 2,048-deep contraction at 18.6 TFLOP/s and a 4,096-deep
//! one at 9.7 (shape sweep, issue #161), so a deeper projection is cut into summed
//! parts. A part must also divide every contraction depth in one program, because each
//! stage splits its input evenly.

/// The deepest contraction measured at the Neural Engine's full rate.
const DEEPEST_FULL_RATE_CONTRACTION: usize = 2048;
/// Every part is its own convolution and summing step. Below this, a shape whose depths
/// barely share a divisor would write thousands of operations per program, so it is
/// refused and the model stays on the GPU.
const SHALLOWEST_SENSIBLE_CONTRACTION: usize = 64;

const fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

/// The deepest part that divides every depth and runs at the full rate, or `None` when
/// a depth is zero or no sensible part exists. Deeper parts need fewer summing
/// operations, so the deepest one wins.
#[must_use]
pub fn contraction_part_width(contraction_depths: &[usize]) -> Option<usize> {
    let mut shared_depth = 0;
    for contraction_depth in contraction_depths {
        if *contraction_depth == 0 {
            return None;
        }
        shared_depth = greatest_common_divisor(shared_depth, *contraction_depth);
    }
    if shared_depth == 0 {
        return None;
    }
    let mut part_count = 1;
    while shared_depth / part_count > DEEPEST_FULL_RATE_CONTRACTION
        || !shared_depth.is_multiple_of(part_count)
    {
        part_count += 1;
    }
    let part_width = shared_depth / part_count;
    (part_width >= SHALLOWEST_SENSIBLE_CONTRACTION).then_some(part_width)
}

#[cfg(test)]
#[path = "contraction_split_tests.rs"]
mod tests;
