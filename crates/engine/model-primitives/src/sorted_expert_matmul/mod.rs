//! Sorted-expert 4-bit affine matmul on M5 matrix units: the multiply a
//! mixture-of-experts layer makes for a prompt chunk, after its rows are
//! sorted by expert, with the same bits as MLX's `gather_qmm` and laid out
//! to read each row and weight block fewer times (see the `metal` folder's
//! `astro_primitive_sorted_expert_matmul_body.metal`).

pub mod dispatch_plan;
#[cfg(feature = "direct-mlx")]
pub mod sorted_expert_quantized_matmul;

#[cfg(test)]
mod dispatch_plan_tests;

pub use dispatch_plan::{
    SortedExpertDispatchPlan, SortedExpertDispatchRefusal, SortedExpertLayout,
};
#[cfg(feature = "direct-mlx")]
pub use sorted_expert_quantized_matmul::{
    ExpertStack, SORTED_EXPERT_LAYOUT, SortedExpertMatmulKernels, SortedExpertProjectionError,
    sorted_expert_quantized_matmul, sorted_expert_quantized_matmul_from_token_rows,
    sorted_expert_quantized_matmul_with_layout,
};
