//! The requantizer writes mlx-lm's layout, not the source's: a source spelled
//! the way an original Qwen3.5 release spells it comes out under mlx-lm's
//! names with MLX convolution axes and materialized norm gammas, its tower in
//! the bfloat16 sidecar, and its prediction head and tied output head gone.

use std::path::Path;

use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::encoder::RangeSearchPolicy;
use astronomical_quantization::module_inventory::ModuleInventory;
use astronomical_quantization::progress::SilentProgressReporter;
use astronomical_quantization::requantize::bit_policy::{
    BitPolicy, StructuralMixtureOfExpertsPolicy,
};
use astronomical_quantization::requantize::driver::{VISION_SIDECAR_FILE_NAME, run_requantize};
use astronomical_quantization::requantize::tensor_plan::RequantizeTensorPlan;
use astronomical_quantization::requantize::{MTP_SIDECAR_RELATIVE_PATH, RequantizeOutcome};
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};
use serial_test::parallel;

use super::requantize_layout_contracts::{bfloat16_weight, save_source};
use super::support::contract_runtime;

const HIDDEN_COLUMNS: usize = 128;
const SHARD_BYTES_LIMIT: u64 = 64 * 1024;
const MODEL_SHARD_FILE_NAME: &str = "model.safetensors";

const VISION_QWEN_CONFIG: &str = r#"{
    "model_type": "qwen3_5",
    "vision_config": {"depth": 1},
    "text_config": {"model_type": "qwen3_5_text", "tie_word_embeddings": true}
}"#;

/// Writes the hub-spelled source, requantizes it, and returns the output
/// with the requantizer's own summary.
fn requantized_hub_release(runtime: &MlxRuntime) -> (tempfile::TempDir, RequantizeOutcome) {
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    let decay_values = [0.5_f32; 16];
    let decay_parameter = runtime
        .array_from_f32(&decay_values, &[16])
        .expect("the decay parameter should be created");
    save_source(
        runtime,
        source.path(),
        &[
            (
                "model.language_model.layers.0.mlp.up_proj.weight",
                &bfloat16_weight(runtime, &[64, HIDDEN_COLUMNS]),
            ),
            (
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                &bfloat16_weight(runtime, &[8, 1, 4]),
            ),
            (
                "model.language_model.layers.0.input_layernorm.weight",
                &bfloat16_weight(runtime, &[HIDDEN_COLUMNS]),
            ),
            (
                "model.language_model.layers.0.linear_attn.A_log",
                &decay_parameter,
            ),
            (
                "model.visual.patch_embed.proj.weight",
                &bfloat16_weight(runtime, &[8, 3, 2, 4, 4]),
            ),
            (
                "mtp.fc.weight",
                &bfloat16_weight(runtime, &[64, HIDDEN_COLUMNS]),
            ),
            (
                "lm_head.weight",
                &bfloat16_weight(runtime, &[64, HIDDEN_COLUMNS]),
            ),
        ],
        VISION_QWEN_CONFIG,
    );
    let inventory =
        ModuleInventory::from_source_directory(source.path(), GroupSize::new(64).expect("valid"))
            .expect("the inventory should build");
    let plan = RequantizeTensorPlan::from_inventory(
        &inventory,
        &BitPolicy::Uniform {
            bit_width: BitWidth::new(4).expect("valid"),
            group_size: GroupSize::new(64).expect("valid"),
        },
    );
    let outcome = run_requantize(
        runtime,
        source.path(),
        output.path(),
        &plan,
        RangeSearchPolicy::default_policy(),
        SHARD_BYTES_LIMIT,
        &mut SilentProgressReporter,
    )
    .expect("requantization should succeed");
    (output, outcome)
}

fn weight_map(output: &Path) -> serde_json::Value {
    let index_text = std::fs::read_to_string(output.join("model.safetensors.index.json"))
        .expect("the index should be readable");
    let index_document: serde_json::Value =
        serde_json::from_str(&index_text).expect("the index should parse");
    index_document["weight_map"].clone()
}

fn written_tensor(runtime: &MlxRuntime, output: &Path, file_name: &str, key: &str) -> MlxArray {
    let stored_file = std::fs::File::open(output.join(file_name)).expect("the file should open");
    runtime
        .load_safetensors(stored_file, None)
        .expect("the file should load")
        .tensor(key)
        .expect("the tensor should be in the file")
}

#[test]
#[parallel]
fn should_write_language_tensors_under_mlx_lm_names_and_drop_the_heads_it_drops() {
    let runtime = contract_runtime();

    let (output, _) = requantized_hub_release(&runtime);

    let weight_map = weight_map(output.path());
    let written_names = weight_map.as_object().expect("the weight map is an object");
    assert_eq!(
        weight_map["language_model.model.layers.0.mlp.up_proj.scales"],
        MODEL_SHARD_FILE_NAME
    );
    assert_eq!(
        weight_map["language_model.model.layers.0.input_layernorm.weight"],
        MODEL_SHARD_FILE_NAME
    );
    for written_name in written_names.keys() {
        assert!(!written_name.starts_with("model."), "{written_name}");
        assert!(!written_name.contains("mtp."), "{written_name}");
        assert!(!written_name.contains("lm_head"), "{written_name}");
    }
}

#[test]
#[parallel]
fn should_move_the_convolution_axes_and_materialize_the_norm_gamma() {
    let runtime = contract_runtime();
    let source_norm = bfloat16_weight(&runtime, &[HIDDEN_COLUMNS]);

    let (output, _) = requantized_hub_release(&runtime);

    let convolution = written_tensor(
        &runtime,
        output.path(),
        MODEL_SHARD_FILE_NAME,
        "language_model.model.layers.0.linear_attn.conv1d.weight",
    );
    let written_norm = written_tensor(
        &runtime,
        output.path(),
        MODEL_SHARD_FILE_NAME,
        "language_model.model.layers.0.input_layernorm.weight",
    );
    let one = runtime
        .astype(&MlxArray::from_scalar_f32(1.0), MlxDtype::BFloat16)
        .expect("cast");
    let expected_norm = runtime.add(&source_norm, &one).expect("the offset adds");
    assert_eq!(convolution.shape(), vec![8, 4, 1]);
    assert_eq!(written_norm.dtype(), MlxDtype::BFloat16);
    assert_eq!(
        float32_values(&runtime, &written_norm),
        float32_values(&runtime, &expected_norm)
    );
}

#[test]
#[parallel]
fn should_keep_the_decay_parameter_at_its_stored_width() {
    let runtime = contract_runtime();

    let (output, _) = requantized_hub_release(&runtime);

    let decay_parameter = written_tensor(
        &runtime,
        output.path(),
        MODEL_SHARD_FILE_NAME,
        "language_model.model.layers.0.linear_attn.A_log",
    );
    assert_eq!(decay_parameter.dtype(), MlxDtype::Float32);
    assert_eq!(float32_values(&runtime, &decay_parameter), vec![0.5; 16]);
}

#[test]
#[parallel]
fn should_write_the_tower_to_the_sidecar_channel_last_and_list_it_in_the_index() {
    let runtime = contract_runtime();

    let (output, outcome) = requantized_hub_release(&runtime);

    let patch_embedding = written_tensor(
        &runtime,
        output.path(),
        VISION_SIDECAR_FILE_NAME,
        "vision_tower.patch_embed.proj.weight",
    );
    assert_eq!(outcome.vision_sidecar_tensor_count, 1);
    assert_eq!(patch_embedding.shape(), vec![8, 2, 4, 4, 3]);
    assert_eq!(patch_embedding.dtype(), MlxDtype::BFloat16);
    assert_eq!(
        weight_map(output.path())["vision_tower.patch_embed.proj.weight"],
        VISION_SIDECAR_FILE_NAME
    );
}

#[test]
#[parallel]
fn should_write_no_sidecar_for_a_source_without_a_tower() {
    let runtime = contract_runtime();
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    save_source(
        &runtime,
        source.path(),
        &[(
            "model.language_model.layers.0.mlp.up_proj.weight",
            &bfloat16_weight(&runtime, &[64, HIDDEN_COLUMNS]),
        )],
        VISION_QWEN_CONFIG,
    );
    let inventory =
        ModuleInventory::from_source_directory(source.path(), GroupSize::new(64).expect("valid"))
            .expect("the inventory should build");
    let plan = RequantizeTensorPlan::from_inventory(
        &inventory,
        &BitPolicy::Uniform {
            bit_width: BitWidth::new(4).expect("valid"),
            group_size: GroupSize::new(64).expect("valid"),
        },
    );

    let outcome = run_requantize(
        &runtime,
        source.path(),
        output.path(),
        &plan,
        RangeSearchPolicy::default_policy(),
        SHARD_BYTES_LIMIT,
        &mut SilentProgressReporter,
    )
    .expect("requantization should succeed");

    assert_eq!(outcome.vision_sidecar_tensor_count, 0);
    assert!(!output.path().join(VISION_SIDECAR_FILE_NAME).exists());
    assert!(inventory.declares_multimodal_towers());
}

fn float32_values(runtime: &MlxRuntime, tensor: &MlxArray) -> Vec<f32> {
    runtime
        .astype(tensor, MlxDtype::Float32)
        .expect("cast")
        .to_vec_f32()
        .expect("values read back")
}

/// A hub-spelled source with a prediction head: a projection, a norm, the
/// fusion layer, and one fused routed-expert tensor of two experts.
fn requantized_release_with_a_prediction_head(
    runtime: &MlxRuntime,
    policy: &BitPolicy,
) -> (tempfile::TempDir, RequantizeOutcome, MlxArray) {
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    let fused_experts = bfloat16_weight(runtime, &[2, 128, HIDDEN_COLUMNS]);
    save_source(
        runtime,
        source.path(),
        &[
            (
                "model.language_model.layers.0.mlp.up_proj.weight",
                &bfloat16_weight(runtime, &[64, HIDDEN_COLUMNS]),
            ),
            (
                "mtp.layers.0.self_attn.q_proj.weight",
                &bfloat16_weight(runtime, &[64, HIDDEN_COLUMNS]),
            ),
            ("mtp.layers.0.mlp.experts.gate_up_proj", &fused_experts),
            (
                "mtp.norm.weight",
                &bfloat16_weight(runtime, &[HIDDEN_COLUMNS]),
            ),
            (
                "mtp.fc.weight",
                &bfloat16_weight(runtime, &[64, HIDDEN_COLUMNS]),
            ),
        ],
        VISION_QWEN_CONFIG,
    );
    let inventory =
        ModuleInventory::from_source_directory(source.path(), GroupSize::new(64).expect("valid"))
            .expect("the inventory should build");
    let plan = RequantizeTensorPlan::from_inventory(&inventory, policy);
    let outcome = run_requantize(
        runtime,
        source.path(),
        output.path(),
        &plan,
        RangeSearchPolicy::default_policy(),
        SHARD_BYTES_LIMIT,
        &mut SilentProgressReporter,
    )
    .expect("requantization should succeed");
    (output, outcome, fused_experts)
}

fn final_conversion_policy() -> BitPolicy {
    BitPolicy::Structural(StructuralMixtureOfExpertsPolicy::optiq_defaults())
}

#[test]
#[parallel]
fn should_write_the_prediction_head_quantized_beside_its_untouched_norm_and_fusion_layer() {
    let runtime = contract_runtime();

    let (output, outcome, _) =
        requantized_release_with_a_prediction_head(&runtime, &final_conversion_policy());

    let head = outcome.mtp_sidecar.expect("the head should be written");
    let query_words = written_tensor(
        &runtime,
        output.path(),
        MTP_SIDECAR_RELATIVE_PATH,
        "mtp.layers.0.self_attn.q_proj.weight",
    );
    let fusion_layer = written_tensor(
        &runtime,
        output.path(),
        MTP_SIDECAR_RELATIVE_PATH,
        "mtp.fc.weight",
    );
    let norm = written_tensor(
        &runtime,
        output.path(),
        MTP_SIDECAR_RELATIVE_PATH,
        "mtp.norm.weight",
    );
    // One projection and four expert slices at three tensors each, plus the
    // norm and the fusion layer.
    assert_eq!(head.tensor_count, 3 + 4 * 3 + 2);
    assert_eq!((head.bits, head.group_size, head.layer_count), (4, 64, 1));
    assert_eq!(query_words.dtype(), MlxDtype::UInt32);
    assert_eq!(
        query_words.shape(),
        vec![64, HIDDEN_COLUMNS as i32 * 4 / 32]
    );
    assert_eq!(fusion_layer.dtype(), MlxDtype::BFloat16);
    assert_eq!(fusion_layer.shape(), vec![64, HIDDEN_COLUMNS as i32]);
    assert_eq!(norm.shape(), vec![HIDDEN_COLUMNS as i32]);
    for written_name in weight_map(output.path())
        .as_object()
        .expect("the weight map is an object")
        .keys()
    {
        assert!(
            !written_name.contains("mtp."),
            "{written_name} is in the model index"
        );
    }
}

#[test]
#[parallel]
fn should_split_a_fused_head_expert_tensor_into_per_expert_quantized_projections() {
    let runtime = contract_runtime();

    let (output, _, fused_experts) =
        requantized_release_with_a_prediction_head(&runtime, &final_conversion_policy());

    // The second expert's up projection is the upper half of its rows.
    let second_expert_up = runtime
        .reshape(
            &runtime
                .slice(
                    &fused_experts,
                    &[1, 64, 0],
                    &[2, 128, HIDDEN_COLUMNS as i32],
                    &[1, 1, 1],
                )
                .expect("the slice should be taken"),
            &[64, HIDDEN_COLUMNS as i32],
        )
        .expect("the slice should reshape");
    let (expected_words, _, _) = runtime
        .quantize_affine(&second_expert_up, 64, 4)
        .expect("stock quantization should succeed");
    let written_words = written_tensor(
        &runtime,
        output.path(),
        MTP_SIDECAR_RELATIVE_PATH,
        "mtp.layers.0.mlp.experts.1.up_proj.weight",
    );
    assert_eq!(
        written_words.to_vec_u32().expect("words read back"),
        expected_words.to_vec_u32().expect("words read back")
    );
}

#[test]
#[parallel]
fn should_write_no_prediction_head_for_a_baseline_built_to_be_probed() {
    let runtime = contract_runtime();
    let uniform_baseline = BitPolicy::Uniform {
        bit_width: BitWidth::new(4).expect("valid"),
        group_size: GroupSize::new(64).expect("valid"),
    };

    let (output, outcome, _) =
        requantized_release_with_a_prediction_head(&runtime, &uniform_baseline);

    assert!(outcome.mtp_sidecar.is_none());
    assert!(!output.path().join(MTP_SIDECAR_RELATIVE_PATH).exists());
}
