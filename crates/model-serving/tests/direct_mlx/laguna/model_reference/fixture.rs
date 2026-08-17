//! Deterministic contracts and canonical weights for complete-model references.
//!
//! The compact rows retain the architectural distinctions that can alter Laguna
//! execution while shrinking dimensions that only multiply test cost. This
//! keeps the matrix adaptive to the available Mac instead of turning a real
//! XS or S allocation into a prerequisite for ordinary direct-MLX verification.

use std::collections::HashMap;

use astronomical_model_serving::{
    LagunaAttentionProjection, LagunaExpertProjection, LagunaGlobalTensorRole,
    LagunaLayerTensorRole, LagunaModel, LagunaNativeWeights, LagunaTargetNormalizer,
    LagunaTensorComponent, LagunaTensorId,
};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};
use serde_json::{Value, json};

pub(super) struct ReferenceRow {
    pub(super) row_name: &'static str,
    pub(super) target_config: Value,
    pub(super) activation_dtype: MlxDtype,
    pub(super) prefill_token_ids: Vec<u32>,
    pub(super) decode_token_ids: Vec<u32>,
    pub(super) tolerance: f32,
    affine_profile: Option<ReferenceAffineProfile>,
    pub(super) has_attention_gate: bool,
    pub(super) has_sliding_attention: bool,
}

#[derive(Clone, Copy)]
struct ReferenceAffineProfile {
    bits: i32,
    group_size: i32,
}

pub(super) struct ReferenceFixture {
    pub(super) model: LagunaModel,
    pub(super) reference_tensors: HashMap<LagunaTensorId, MlxArray>,
}

pub(super) fn generic_rows() -> Vec<ReferenceRow> {
    // The three rows separate topology, gating granularity, output tying, and
    // activation dtype so one passing mixed row cannot hide an unused branch.
    vec![
        row(
            "generic_full",
            &["full"],
            &["none"],
            &[4],
            "float32",
            true,
            4,
        ),
        row(
            "generic_sliding",
            &["sliding"],
            &["per_element"],
            &[4],
            "float16",
            false,
            2,
        ),
        row(
            "generic_mixed",
            &["sliding", "full", "sliding"],
            &["none", "per_head", "per_element"],
            &[4, 6, 4],
            "bfloat16",
            false,
            2,
        ),
    ]
}

pub(super) fn named_rows() -> Vec<ReferenceRow> {
    vec![
        named_row(
            "xs_compact",
            40,
            64,
            "bfloat16",
            ReferenceAffineProfile {
                bits: 2,
                group_size: 64,
            },
            32.0,
            64.0,
            1.346_573_6,
        ),
        named_row(
            "s_compact",
            48,
            72,
            "float16",
            ReferenceAffineProfile {
                bits: 2,
                group_size: 128,
            },
            128.0,
            32.0,
            1.485_203,
        ),
    ]
}

pub(super) fn build_fixture(runtime: &MlxRuntime, row: &ReferenceRow) -> ReferenceFixture {
    let contract = LagunaTargetNormalizer::normalize(
        &serde_json::to_vec(&row.target_config).expect("reference config should serialize"),
    )
    .unwrap_or_else(|error| panic!("{} should normalize: {error}", row.row_name));
    let (production_tensors, reference_tensors) =
        tensor_inventories(runtime, &contract, row.activation_dtype, row.affine_profile);
    let weights = LagunaNativeWeights::bind(runtime, production_tensors, &contract)
        .unwrap_or_else(|error| panic!("{} weights should bind: {error:?}", row.row_name));
    let model = LagunaModel::new(contract, weights)
        .unwrap_or_else(|error| panic!("{} model should construct: {error:?}", row.row_name));
    ReferenceFixture {
        model,
        reference_tensors,
    }
}

fn row(
    row_name: &'static str,
    layer_types: &[&str],
    gating_types: &[&str],
    query_head_counts: &[u32],
    dtype_name: &str,
    has_tied_embeddings: bool,
    sliding_window: u32,
) -> ReferenceRow {
    let config = base_config(
        layer_types.len(),
        layer_types,
        gating_types,
        query_head_counts,
        dtype_name,
        has_tied_embeddings,
        sliding_window,
        2,
        16,
    );
    ReferenceRow {
        row_name,
        target_config: config,
        activation_dtype: dtype(dtype_name),
        prefill_token_ids: vec![1, 2],
        decode_token_ids: vec![3],
        // This is the established primitive-level low-precision tolerance; the
        // generic rows are short enough that layer accumulation needs no relief.
        tolerance: if dtype_name == "float32" { 5e-4 } else { 2e-2 },
        affine_profile: None,
        has_attention_gate: gating_types.iter().any(|gating| *gating != "none"),
        has_sliding_attention: layer_types.contains(&"sliding"),
    }
}

fn named_row(
    row_name: &'static str,
    layer_count: usize,
    sliding_query_heads: u32,
    dtype_name: &str,
    affine_profile: ReferenceAffineProfile,
    yarn_factor: f64,
    yarn_beta_fast: f64,
    yarn_attention_factor: f64,
) -> ReferenceRow {
    let layer_types = (0..layer_count)
        .map(|layer_index| {
            if layer_index % 4 == 0 {
                "full"
            } else {
                "sliding"
            }
        })
        .collect::<Vec<_>>();
    let gating_types = vec!["per_head"; layer_count];
    let query_heads = layer_types
        .iter()
        .map(|layer_type| {
            if *layer_type == "full" {
                48
            } else {
                sliding_query_heads
            }
        })
        .collect::<Vec<_>>();
    let mut config = base_config(
        layer_count,
        &layer_types,
        &gating_types,
        &query_heads,
        dtype_name,
        false,
        512,
        128,
        affine_profile.group_size,
    );
    config["num_key_value_heads"] = json!(8);
    config["quantization"] = json!({
        "bits": affine_profile.bits,
        "group_size": affine_profile.group_size,
        "mode": "affine"
    });
    config["quantization_config"] = config["quantization"].clone();
    config["rope_parameters"] = json!({
        "full_attention": {
            "rope_type": "yarn", "rope_theta": 500000.0, "factor": yarn_factor,
            "original_max_position_embeddings": 8192, "beta_slow": 1.0,
            "beta_fast": yarn_beta_fast, "attention_factor": yarn_attention_factor,
            "partial_rotary_factor": 0.5
        },
        "sliding_attention": {
            "rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0
        }
    });
    ReferenceRow {
        row_name,
        target_config: config,
        activation_dtype: dtype(dtype_name),
        prefill_token_ids: vec![1],
        decode_token_ids: vec![2],
        // Quantized MLX accumulation and the dequantized dense oracle associate
        // products differently. One extra percentage point above the primitive
        // tolerance covers that difference across 40/48 layers without making
        // descriptor-order or residual-placement errors acceptable.
        tolerance: 3e-2,
        affine_profile: Some(affine_profile),
        has_attention_gate: true,
        has_sliding_attention: true,
    }
}

#[allow(clippy::too_many_arguments)]
fn base_config(
    layer_count: usize,
    layer_types: &[&str],
    gating_types: &[&str],
    query_head_counts: &[u32],
    dtype_name: &str,
    tied: bool,
    sliding_window: u32,
    head_dimension: u32,
    hidden_size: i32,
) -> Value {
    json!({
        "architectures": ["LagunaForCausalLM"], "model_type": "laguna",
        "vocab_size": 8, "hidden_size": hidden_size, "intermediate_size": hidden_size,
        "num_hidden_layers": layer_count, "num_attention_heads": query_head_counts[0],
        "num_attention_heads_per_layer": query_head_counts,
        "num_key_value_heads": 2, "head_dim": head_dimension,
        "max_position_embeddings": 32768, "rms_norm_eps": 0.00001,
        "tie_word_embeddings": tied, "torch_dtype": dtype_name,
        "layer_types": layer_types, "sliding_window": sliding_window,
        "mlp_layer_types": vec!["dense"; layer_count], "gating_types": gating_types,
        "rope_parameters": { "rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0 }
    })
}

fn tensor_inventories(
    runtime: &MlxRuntime,
    contract: &astronomical_model_serving::LagunaTargetContract,
    dtype: MlxDtype,
    affine_profile: Option<ReferenceAffineProfile>,
) -> (
    HashMap<LagunaTensorId, MlxArray>,
    HashMap<LagunaTensorId, MlxArray>,
) {
    let hidden_size = contract.model().hidden_size() as i32;
    let vocabulary_size = contract.model().vocabulary_size() as i32;
    // Production receives packed affine arrays while the oracle receives the
    // exact MLX-dequantized matrices. Sharing the pre-quantized source here
    // would accidentally compare two dense paths and leave qmm unqualified.
    let mut production_tensors = HashMap::new();
    let mut reference_tensors = HashMap::new();
    insert_matrix(
        &mut production_tensors,
        &mut reference_tensors,
        runtime,
        global(LagunaGlobalTensorRole::TokenEmbedding),
        &[vocabulary_size, hidden_size],
        dtype,
        3,
        affine_profile,
    );
    insert_vector(
        &mut production_tensors,
        &mut reference_tensors,
        global(LagunaGlobalTensorRole::FinalNormalization),
        norm(runtime, hidden_size, dtype, 5),
    );
    if !contract.model().has_tied_embeddings() {
        insert_matrix(
            &mut production_tensors,
            &mut reference_tensors,
            runtime,
            global(LagunaGlobalTensorRole::OutputHead),
            &[vocabulary_size, hidden_size],
            dtype,
            7,
            affine_profile,
        );
    }
    for layer in contract.layers() {
        let layer_index = layer.layer_index();
        let attention = layer.attention();
        let query_width = attention.query_head_count() as i32 * attention.head_dimension() as i32;
        let key_value_width =
            attention.key_value_head_count() as i32 * attention.head_dimension() as i32;
        insert_vector(
            &mut production_tensors,
            &mut reference_tensors,
            layer_id(layer_index, LagunaLayerTensorRole::InputNormalization),
            norm(runtime, hidden_size, dtype, 11 + layer_index),
        );
        insert_vector(
            &mut production_tensors,
            &mut reference_tensors,
            layer_id(
                layer_index,
                LagunaLayerTensorRole::PostAttentionNormalization,
            ),
            norm(runtime, hidden_size, dtype, 17 + layer_index),
        );
        insert_vector(
            &mut production_tensors,
            &mut reference_tensors,
            layer_id(
                layer_index,
                LagunaLayerTensorRole::AttentionQueryNormalization,
            ),
            norm(
                runtime,
                attention.head_dimension() as i32,
                dtype,
                23 + layer_index,
            ),
        );
        insert_vector(
            &mut production_tensors,
            &mut reference_tensors,
            layer_id(
                layer_index,
                LagunaLayerTensorRole::AttentionKeyNormalization,
            ),
            norm(
                runtime,
                attention.head_dimension() as i32,
                dtype,
                29 + layer_index,
            ),
        );
        for (projection, output_width, seed) in [
            (LagunaAttentionProjection::Query, query_width, 31),
            (LagunaAttentionProjection::Key, key_value_width, 37),
            (LagunaAttentionProjection::Value, key_value_width, 41),
            (LagunaAttentionProjection::Output, hidden_size, 43),
        ] {
            let input_width = if projection == LagunaAttentionProjection::Output {
                query_width
            } else {
                hidden_size
            };
            insert_matrix(
                &mut production_tensors,
                &mut reference_tensors,
                runtime,
                layer_id(layer_index, LagunaLayerTensorRole::Attention(projection)),
                &[output_width, input_width],
                dtype,
                seed + layer_index,
                affine_profile,
            );
        }
        let gate_width = match attention.gating_kind() {
            astronomical_model_serving::LagunaGatingKind::None => 0,
            astronomical_model_serving::LagunaGatingKind::PerHead => {
                attention.query_head_count() as i32
            }
            astronomical_model_serving::LagunaGatingKind::PerElement => query_width,
        };
        if gate_width > 0 {
            insert_matrix(
                &mut production_tensors,
                &mut reference_tensors,
                runtime,
                layer_id(
                    layer_index,
                    LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Gate),
                ),
                &[gate_width, hidden_size],
                dtype,
                47 + layer_index,
                affine_profile,
            );
        }
        let intermediate_size = match layer.feed_forward() {
            astronomical_model_serving::LagunaFeedForwardDescriptor::Dense(descriptor) => {
                descriptor.intermediate_size() as i32
            }
            _ => unreachable!("the issue #101 reference rows remain dense"),
        };
        for (projection, shape, seed) in [
            (
                LagunaExpertProjection::Gate,
                vec![intermediate_size, hidden_size],
                53,
            ),
            (
                LagunaExpertProjection::Up,
                vec![intermediate_size, hidden_size],
                59,
            ),
            (
                LagunaExpertProjection::Down,
                vec![hidden_size, intermediate_size],
                61,
            ),
        ] {
            insert_matrix(
                &mut production_tensors,
                &mut reference_tensors,
                runtime,
                layer_id(
                    layer_index,
                    LagunaLayerTensorRole::DenseFeedForward(projection),
                ),
                &shape,
                dtype,
                seed + layer_index,
                affine_profile,
            );
        }
    }
    (production_tensors, reference_tensors)
}

fn deterministic(runtime: &MlxRuntime, shape: &[i32], dtype: MlxDtype, seed: usize) -> MlxArray {
    let element_count = shape.iter().product::<i32>() as usize;
    // A short signed period makes every projection non-constant while keeping
    // accumulated logits bounded enough for meaningful low-precision tolerances.
    let values = (0..element_count)
        .map(|element_index| (((element_index + seed) % 19) as f32 - 9.0) / 64.0)
        .collect::<Vec<_>>();
    runtime
        .array_from_f32(&values, shape)
        .and_then(|array| runtime.astype(&array, dtype))
        .expect("deterministic Laguna weight should construct")
}

fn norm(runtime: &MlxRuntime, width: i32, dtype: MlxDtype, seed: usize) -> MlxArray {
    let values = (0..width)
        .map(|index| 0.9 + ((index as usize + seed) % 5) as f32 * 0.025)
        .collect::<Vec<_>>();
    runtime
        .array_from_f32(&values, &[width])
        .and_then(|array| runtime.astype(&array, dtype))
        .expect("normalization weight should construct")
}

fn global(role: LagunaGlobalTensorRole) -> LagunaTensorId {
    LagunaTensorId::Global {
        role,
        component: LagunaTensorComponent::Weight,
    }
}

fn layer_id(layer_index: usize, role: LagunaLayerTensorRole) -> LagunaTensorId {
    LagunaTensorId::Layer {
        layer_index,
        role,
        component: LagunaTensorComponent::Weight,
    }
}

#[allow(clippy::too_many_arguments)]
fn insert_matrix(
    production_tensors: &mut HashMap<LagunaTensorId, MlxArray>,
    reference_tensors: &mut HashMap<LagunaTensorId, MlxArray>,
    runtime: &MlxRuntime,
    tensor_id: LagunaTensorId,
    shape: &[i32],
    dtype: MlxDtype,
    seed: usize,
    affine_profile: Option<ReferenceAffineProfile>,
) {
    let source_weight = deterministic(runtime, shape, dtype, seed);
    let reference_weight = if let Some(affine_profile) = affine_profile {
        let (packed_weight, scales, biases) = runtime
            .quantize_affine(
                &source_weight,
                affine_profile.group_size,
                affine_profile.bits,
            )
            .expect("affine reference weight should quantize");
        let dequantized_weight = runtime
            .dequantize_affine(
                &packed_weight,
                &scales,
                &biases,
                affine_profile.group_size,
                affine_profile.bits,
            )
            .expect("affine reference weight should dequantize");
        assert!(
            production_tensors
                .insert(tensor_id, packed_weight)
                .is_none()
        );
        assert!(
            production_tensors
                .insert(
                    with_component(tensor_id, LagunaTensorComponent::Scales),
                    scales
                )
                .is_none()
        );
        assert!(
            production_tensors
                .insert(
                    with_component(tensor_id, LagunaTensorComponent::Biases),
                    biases
                )
                .is_none()
        );
        dequantized_weight
    } else {
        let retained_weight = source_weight
            .retain()
            .expect("native production weight should retain");
        assert!(
            production_tensors
                .insert(tensor_id, retained_weight)
                .is_none()
        );
        source_weight
    };
    assert!(
        reference_tensors
            .insert(tensor_id, reference_weight)
            .is_none()
    );
}

fn insert_vector(
    production_tensors: &mut HashMap<LagunaTensorId, MlxArray>,
    reference_tensors: &mut HashMap<LagunaTensorId, MlxArray>,
    tensor_id: LagunaTensorId,
    tensor: MlxArray,
) {
    let retained_tensor = tensor.retain().expect("production vector should retain");
    assert!(
        production_tensors
            .insert(tensor_id, retained_tensor)
            .is_none()
    );
    assert!(reference_tensors.insert(tensor_id, tensor).is_none());
}

fn with_component(tensor_id: LagunaTensorId, component: LagunaTensorComponent) -> LagunaTensorId {
    match tensor_id {
        LagunaTensorId::Global { role, .. } => LagunaTensorId::Global { role, component },
        LagunaTensorId::Layer {
            layer_index, role, ..
        } => LagunaTensorId::Layer {
            layer_index,
            role,
            component,
        },
    }
}

fn dtype(dtype_name: &str) -> MlxDtype {
    match dtype_name {
        "float32" => MlxDtype::Float32,
        "float16" => MlxDtype::Float16,
        "bfloat16" => MlxDtype::BFloat16,
        _ => unreachable!("reference dtype is fixed by the row"),
    }
}
