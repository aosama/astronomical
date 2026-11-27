// One pass that reduces sorted routed-expert outputs and adds the gated
// shared expert, keeping the sparse sum in a register instead of writing the
// [tokens, hidden] sum and reading it back in a second launch.
//
// It stays bit for bit equal to the two-step path (the sorted weighted sum,
// then MLX's compiled sigmoid-multiply-add graph), so every intermediate is
// rounded to bfloat16 exactly where the two-step path stores it: the Float32
// weighted sum (the shared reduction prologue) becomes bfloat16 before the
// add, the sigmoid is MLX's own Sigmoid expression evaluated on bfloat16, and
// the multiply and add each run on bfloat16 operands.
//
// The sigmoid lines copy MLX's Sigmoid::operator() (metal/kernels/unary_ops.h)
// on purpose: any other formulation, even a mathematically equal one, rounds
// differently in bfloat16 and breaks the bit-for-bit contract with the
// compiled graph. sigmoid_of_negative_magnitude is sigmoid(-|x|): the gate
// itself for a negative logit and its complement for a non-negative one.
//
// @prologue astro_primitive_sorted_expert_weighted_reduction_body.metal
// @input sorted_outputs OutputT
// @input inverse_order uint32_t may-be-small
// @input scores bfloat16_t|float16_t|float may-be-small
// @input shared_expert_output OutputT
// @input shared_expert_gate_logits OutputT may-be-small
// @output combined_expert_output OutputT
// @template OutputT typename
// @template selected_expert_count int
// @template output_dimension int
// @compile-with OutputT=bfloat16_t selected_expert_count=8 output_dimension=2048
// @compile-with OutputT=bfloat16_t selected_expert_count=1 output_dimension=4
// @contract qwen35 qwen3_5_core::model_math::sorted_expert_shared_combination_tests
// @probe sorted_expert_shared_combination

    OutputT sparse_expert_output = (OutputT)weighted_sum;
    OutputT gate_logit = shared_expert_gate_logits[token_index];
    auto sigmoid_of_negative_magnitude =
        1 / (1 + metal::precise::exp(metal::abs(gate_logit)));
    OutputT shared_expert_gate = (gate_logit < 0)
        ? sigmoid_of_negative_magnitude
        : 1 - sigmoid_of_negative_magnitude;
    OutputT gated_shared_expert_output =
        shared_expert_output[output_index] * shared_expert_gate;
    combined_expert_output[output_index] =
        sparse_expert_output + gated_shared_expert_output;
