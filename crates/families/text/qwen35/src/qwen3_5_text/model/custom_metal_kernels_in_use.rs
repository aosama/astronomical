//! Which of Astronomical's own Metal kernels a loaded model holds, for the
//! status report. A kernel is held only when its capability probe passed on
//! this GPU and the model's shape and weights take it, so this is the answer
//! to "which kernels serve this model", where a probe verdict alone is not.

use super::streaming_model::Qwen3_5TextModel;

impl Qwen3_5TextModel {
    /// The held kernels by stable use id, sorted: the probe family id where
    /// the kernel has a probe, otherwise the kernel's own name.
    #[must_use]
    pub fn custom_metal_kernels_in_use(&self) -> Vec<String> {
        let accelerators = &self.base.optional_accelerators;
        let held_kernels = [
            (
                "gated_delta_sequence",
                self.base.gated_delta_kernel.is_some(),
            ),
            (
                "gated_delta_boundary_checkpoint",
                self.base.gated_delta_checkpoint_kernel.is_some(),
            ),
            (
                "gdn_decode_prework",
                self.base.gdn_decode_prework_kernel.is_some(),
            ),
            (
                "sorted_expert_weighted_sum",
                self.base.sorted_expert_weighted_sum_kernel.is_some(),
            ),
            (
                "sorted_expert_shared_combination",
                accelerators
                    .sorted_expert_shared_combination_kernel
                    .is_some(),
            ),
            (
                "prism_hadamard_signed_rotation",
                accelerators.signed_hadamard_rotation_kernel.is_some(),
            ),
            (
                "sorted_expert_matmul",
                accelerators.sorted_expert_matmul_kernels.is_some(),
            ),
            (
                "sorted_expert_tiled_matmul",
                accelerators.sorted_expert_tiled_kernels.is_some(),
            ),
            (
                "row_batched_matmul",
                accelerators.register_ordered_matmul_kernels.is_some(),
            ),
            (
                "fused_jangtq2_gathered_projection",
                self.fused_jangtq2_gathered_projection_kernel.is_some(),
            ),
            (
                "gated_delta_tree",
                accelerators.gated_delta_tree_kernel.is_some(),
            ),
            (
                "gated_delta_replay",
                accelerators.gated_delta_replay_kernel.is_some(),
            ),
            (
                "row_exact_tree_attention",
                accelerators.row_exact_tree_attention_kernels.is_some(),
            ),
            ("lane_mixing", accelerators.lane_mixing_kernels.is_some()),
        ];
        let mut kernels_in_use = Vec::new();
        for (kernel_use_id, is_held) in held_kernels {
            if is_held {
                kernels_in_use.push(kernel_use_id.to_owned());
            }
        }
        kernels_in_use.sort();
        kernels_in_use
    }
}
