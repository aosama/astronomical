// The sorted expert weighted sum: sorted_outputs [assignments, D] in sort
// order, gathered back through inverse_order [tokens * K] and weighted by the
// original scores [tokens * K], stored as weighted_outputs [tokens, D] in the
// output dtype. It never builds the expanded [tokens, K, D] tensor.
//
// @prologue astro_primitive_sorted_expert_weighted_reduction_body.metal
// @input sorted_outputs OutputT
// @input inverse_order uint32_t|int32_t may-be-small
// @input scores float|bfloat16_t may-be-small
// @output weighted_outputs OutputT
// @template OutputT typename
// @template selected_expert_count int
// @template output_dimension int
// @compile-with OutputT=bfloat16_t selected_expert_count=8 output_dimension=2048
// @compile-with OutputT=float selected_expert_count=1 output_dimension=4
// @contract direct_mlx::sparse_experts
// @probe sorted_expert_weighted_sum

    weighted_outputs[output_index] = (OutputT)weighted_sum;
