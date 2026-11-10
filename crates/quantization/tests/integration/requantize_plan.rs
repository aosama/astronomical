//! Plan construction against written shards: decisions, pass-through, and
//! output byte estimates, with no tensor data read.

use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::module_inventory::ModuleInventory;
use astronomical_quantization::requantize::bit_policy::{BitPolicy, TensorEncodingDecision};
use astronomical_quantization::requantize::tensor_plan::{RequantizeTensorPlan, TensorPlanEntry};

use super::module_inventory_journeys::write_bf16_source;

#[test]
fn should_plan_uniform_four_bit_for_linears_eight_for_the_router_and_pass_norms_through() {
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
    assert_eq!(plan.quantized_tensor_count(), 2);
    // Qwen's router is a plain linear layer, written under mlx-lm's name at
    // mlx-lm's own width for a mixture-of-experts gate.
    let router = planned_entry(&plan, "language_model.model.layers.0.mlp.gate.weight")
        .expect("the router is planned");
    assert_eq!(
        router.decision,
        TensorEncodingDecision::Encode {
            bit_width: BitWidth::new(8).expect("valid"),
            group_size: GroupSize::new(64).expect("valid"),
        }
    );
    let norm = planned_entry(
        &plan,
        "language_model.model.layers.0.input_layernorm.weight",
    )
    .expect("the norm is planned");
    assert_eq!(norm.decision, TensorEncodingDecision::PassThrough);
    assert_eq!(
        norm.source_tensor_name,
        "model.layers.0.input_layernorm.weight"
    );
    // 256*128 at 4 bits plus 512 groups of scale and bias; the router's
    // 8*128 at 8 bits plus 16 groups; the norm at bfloat16.
    assert_eq!(
        plan.estimated_output_bytes(),
        16_384 + 2_048 + (8 * 128) + (16 * 4) + (128 * 2)
    );
}

#[test]
fn should_carry_each_packed_module_s_own_source_parameters_into_the_plan() {
    // A mixed source: the up projection was written at 8 bits beside a
    // global 4, and dequantizing it with 4 would fail outright.
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    let packed_tensors: [(&str, safetensors::Dtype, &[usize]); 4] = [
        (
            "model.layers.0.mlp.up_proj.weight",
            safetensors::Dtype::U32,
            &[256, 32],
        ),
        (
            "model.layers.0.mlp.up_proj.scales",
            safetensors::Dtype::BF16,
            &[256, 2],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            safetensors::Dtype::U32,
            &[128, 32],
        ),
        (
            "model.layers.0.mlp.down_proj.scales",
            safetensors::Dtype::BF16,
            &[128, 4],
        ),
    ];
    super::module_inventory_journeys::write_typed_source(
        directory.path(),
        &packed_tensors,
        r#"{"model_type": "qwen3_5", "quantization": {"group_size": 64, "bits": 4,
            "model.layers.0.mlp.up_proj": {"group_size": 64, "bits": 8}}}"#,
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

    // The source config keys its widths by the source's spelling; the plan
    // writes under mlx-lm's.
    let up_projection = planned_entry(&plan, "language_model.model.layers.0.mlp.up_proj.weight")
        .expect("the up projection is planned");
    let down_projection =
        planned_entry(&plan, "language_model.model.layers.0.mlp.down_proj.weight")
            .expect("the down projection is planned");
    assert_eq!(up_projection.source_quantization, Some((8, 64)));
    assert_eq!(down_projection.source_quantization, Some((4, 64)));
    assert_eq!(down_projection.shape, vec![128, 256]);
    assert_eq!(plan.tensor_count(), 2);
}

/// The plan entry for one output key. A plain loop rather than
/// `Iterator::find`, which would need a closure.
fn planned_entry<'plan>(
    plan: &'plan RequantizeTensorPlan,
    tensor_key: &str,
) -> Option<&'plan TensorPlanEntry> {
    let mut matching_entry = None;
    for entry in &plan.entries {
        if entry.tensor_key == tensor_key {
            matching_entry = Some(entry);
            break;
        }
    }
    matching_entry
}
