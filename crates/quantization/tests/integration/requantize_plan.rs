//! Plan construction against written shards: decisions, pass-through, and
//! output byte estimates, with no tensor data read.

use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::module_inventory::ModuleInventory;
use astronomical_quantization::requantize::bit_policy::{BitPolicy, TensorEncodingDecision};
use astronomical_quantization::requantize::tensor_plan::RequantizeTensorPlan;

use super::module_inventory_journeys::write_bf16_source;

#[test]
#[ignore = "pending implementation of RequantizeTensorPlan::from_inventory"]
fn should_plan_uniform_four_bit_for_linears_and_pass_through_for_norms_and_routers() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_bf16_source(
        directory.path(),
        &[
            ("model.layers.0.mlp.up_proj.weight", &[256, 128]),
            ("model.layers.0.mlp.gate.weight", &[8, 128]),
            ("model.layers.0.input_layernorm.weight", &[128]),
        ],
        "qwen3_5_moe",
    );
    let inventory = ModuleInventory::from_source_directory(
        directory.path(),
        GroupSize::new(64).expect("valid"),
    )
    .expect("the inventory should build");

    let plan = RequantizeTensorPlan::from_inventory(
        &inventory,
        &BitPolicy::Uniform {
            bit_width: BitWidth::new(4).expect("valid"),
            group_size: GroupSize::new(64).expect("valid"),
        },
    );

    assert_eq!(plan.tensor_count(), 3);
    assert_eq!(plan.quantized_tensor_count(), 1);
    let router = plan
        .entries
        .iter()
        .find(|entry| entry.tensor_key == "model.layers.0.mlp.gate.weight")
        .expect("the router is planned");
    assert_eq!(router.decision, TensorEncodingDecision::PassThrough);
    // 256*128 at 4 bits plus 512 groups of scale and bias, plus the two
    // pass-through tensors at bfloat16.
    assert_eq!(
        plan.estimated_output_bytes(),
        16_384 + 2_048 + (8 * 128 * 2) + (128 * 2)
    );
}
