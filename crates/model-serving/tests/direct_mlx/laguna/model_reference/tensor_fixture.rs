//! Deterministic canonical tensors for production and independent reference graphs.

use std::collections::{BTreeSet, HashMap};

use astronomical_model_serving::{
    LagunaAttentionProjection, LagunaExpertProjection, LagunaFeedForwardDescriptor,
    LagunaGlobalTensorRole, LagunaLayerTensorRole, LagunaTargetContract, LagunaTensorComponent,
    LagunaTensorId,
};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};

use super::rows::ReferenceRow;
use super::tensor_identity::{affine_profile, global, layer_id, with_component};

pub(super) struct ReferenceTensorInventories {
    pub(super) production_tensors: HashMap<LagunaTensorId, MlxArray>,
    pub(super) reference_tensors: HashMap<LagunaTensorId, MlxArray>,
    pub(super) observed_affine_profiles: BTreeSet<(i32, i32)>,
}

pub(super) fn build_tensor_inventories(
    runtime: &MlxRuntime,
    contract: &LagunaTargetContract,
    row: &ReferenceRow,
) -> ReferenceTensorInventories {
    let hidden_size = contract.model().hidden_size() as i32;
    let vocabulary_size = contract.model().vocabulary_size() as i32;
    let mut inventories = ReferenceTensorInventories {
        production_tensors: HashMap::new(),
        reference_tensors: HashMap::new(),
        observed_affine_profiles: BTreeSet::new(),
    };
    insert_matrix(
        &mut inventories,
        runtime,
        contract,
        global(LagunaGlobalTensorRole::TokenEmbedding),
        &[vocabulary_size, hidden_size],
        row.activation_dtype,
        3,
    );
    insert_vector(
        &mut inventories,
        global(LagunaGlobalTensorRole::FinalNormalization),
        norm(runtime, hidden_size, row.activation_dtype, 5),
    );
    if !contract.model().has_tied_embeddings() {
        insert_matrix(
            &mut inventories,
            runtime,
            contract,
            global(LagunaGlobalTensorRole::OutputHead),
            &[vocabulary_size, hidden_size],
            row.activation_dtype,
            7,
        );
    }
    for layer in contract.layers() {
        insert_attention_tensors(
            &mut inventories,
            runtime,
            contract,
            layer,
            row.activation_dtype,
        );
        match layer.feed_forward() {
            LagunaFeedForwardDescriptor::Dense(descriptor) => insert_dense_tensors(
                &mut inventories,
                runtime,
                contract,
                layer.layer_index(),
                hidden_size,
                descriptor.intermediate_size() as i32,
                row.activation_dtype,
            ),
            LagunaFeedForwardDescriptor::Moe(descriptor) => insert_moe_tensors(
                &mut inventories,
                runtime,
                contract,
                layer.layer_index(),
                hidden_size,
                descriptor,
                row,
            ),
        }
    }
    inventories
}

fn insert_attention_tensors(
    inventories: &mut ReferenceTensorInventories,
    runtime: &MlxRuntime,
    contract: &LagunaTargetContract,
    layer: &astronomical_model_serving::LagunaLayerDescriptor,
    activation_dtype: MlxDtype,
) {
    let layer_index = layer.layer_index();
    let hidden_size = contract.model().hidden_size() as i32;
    let attention = layer.attention();
    let query_width = attention.query_head_count() as i32 * attention.head_dimension() as i32;
    let key_value_width =
        attention.key_value_head_count() as i32 * attention.head_dimension() as i32;
    for (role, seed) in [
        (LagunaLayerTensorRole::InputNormalization, 11),
        (LagunaLayerTensorRole::PostAttentionNormalization, 17),
    ] {
        insert_vector(
            inventories,
            layer_id(layer_index, role),
            norm(runtime, hidden_size, activation_dtype, seed + layer_index),
        );
    }
    for (role, seed) in [
        (LagunaLayerTensorRole::AttentionQueryNormalization, 23),
        (LagunaLayerTensorRole::AttentionKeyNormalization, 29),
    ] {
        insert_vector(
            inventories,
            layer_id(layer_index, role),
            norm(
                runtime,
                attention.head_dimension() as i32,
                activation_dtype,
                seed + layer_index,
            ),
        );
    }
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
            inventories,
            runtime,
            contract,
            layer_id(layer_index, LagunaLayerTensorRole::Attention(projection)),
            &[output_width, input_width],
            activation_dtype,
            seed + layer_index,
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
            inventories,
            runtime,
            contract,
            layer_id(
                layer_index,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Gate),
            ),
            &[gate_width, hidden_size],
            activation_dtype,
            47 + layer_index,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn insert_dense_tensors(
    inventories: &mut ReferenceTensorInventories,
    runtime: &MlxRuntime,
    contract: &LagunaTargetContract,
    layer_index: usize,
    hidden_size: i32,
    intermediate_size: i32,
    activation_dtype: MlxDtype,
) {
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
            inventories,
            runtime,
            contract,
            layer_id(
                layer_index,
                LagunaLayerTensorRole::DenseFeedForward(projection),
            ),
            &shape,
            activation_dtype,
            seed + layer_index,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn insert_moe_tensors(
    inventories: &mut ReferenceTensorInventories,
    runtime: &MlxRuntime,
    contract: &LagunaTargetContract,
    layer_index: usize,
    hidden_size: i32,
    descriptor: &astronomical_model_serving::LagunaMoeDescriptor,
    row: &ReferenceRow,
) {
    let expert_count = descriptor.expert_count() as i32;
    let expert_intermediate_size = descriptor.expert_intermediate_size() as i32;
    insert_matrix(
        inventories,
        runtime,
        contract,
        layer_id(layer_index, LagunaLayerTensorRole::Router),
        &[expert_count, hidden_size],
        row.activation_dtype,
        67 + layer_index,
    );
    if row.has_correction_bias {
        let correction_bias_values = (0..expert_count)
            .map(|expert_index| match expert_index % 4 {
                0 | 1 => 0.2,
                2 => -0.1,
                _ => 0.0,
            })
            .collect::<Vec<_>>();
        let correction_bias = runtime
            .array_from_f32(&correction_bias_values, &[expert_count])
            .expect("router correction bias should construct");
        insert_vector(
            inventories,
            layer_id(layer_index, LagunaLayerTensorRole::RouterCorrectionBias),
            correction_bias,
        );
    }
    for (projection, shape, seed) in [
        (
            LagunaExpertProjection::Gate,
            vec![expert_count, expert_intermediate_size, hidden_size],
            71,
        ),
        (
            LagunaExpertProjection::Up,
            vec![expert_count, expert_intermediate_size, hidden_size],
            73,
        ),
        (
            LagunaExpertProjection::Down,
            vec![expert_count, hidden_size, expert_intermediate_size],
            79,
        ),
    ] {
        insert_matrix(
            inventories,
            runtime,
            contract,
            layer_id(layer_index, LagunaLayerTensorRole::RoutedExpert(projection)),
            &shape,
            row.activation_dtype,
            seed + layer_index,
        );
    }
    let shared_intermediate_size = descriptor.shared_expert_intermediate_size() as i32;
    if shared_intermediate_size == 0 {
        return;
    }
    for (projection, shape, seed) in [
        (
            LagunaExpertProjection::Gate,
            vec![shared_intermediate_size, hidden_size],
            83,
        ),
        (
            LagunaExpertProjection::Up,
            vec![shared_intermediate_size, hidden_size],
            89,
        ),
        (
            LagunaExpertProjection::Down,
            vec![hidden_size, shared_intermediate_size],
            97,
        ),
    ] {
        insert_matrix(
            inventories,
            runtime,
            contract,
            layer_id(layer_index, LagunaLayerTensorRole::SharedExpert(projection)),
            &shape,
            row.activation_dtype,
            seed + layer_index,
        );
    }
}

fn insert_matrix(
    inventories: &mut ReferenceTensorInventories,
    runtime: &MlxRuntime,
    contract: &LagunaTargetContract,
    tensor_id: LagunaTensorId,
    shape: &[i32],
    dtype: MlxDtype,
    seed: usize,
) {
    let source_weight = deterministic(runtime, shape, dtype, seed);
    let reference_weight = match affine_profile(contract, tensor_id) {
        Some((bits, group_size)) => {
            let (packed_weight, scales, biases) = runtime
                .quantize_affine(&source_weight, group_size, bits)
                .expect("affine reference weight should quantize");
            assert_eq!(packed_weight.dtype(), MlxDtype::UInt32);
            let dequantized_weight = runtime
                .dequantize_affine(&packed_weight, &scales, &biases, group_size, bits)
                .expect("affine reference weight should dequantize");
            // Materialize one source at a time so compact 40/48-layer matrices
            // retain final packed/reference arrays instead of every packing graph.
            runtime
                .evaluate_arrays(&[&packed_weight, &scales, &biases, &dequantized_weight])
                .expect("affine fixture arrays should materialize");
            insert_unique(
                &mut inventories.production_tensors,
                tensor_id,
                packed_weight,
            );
            insert_unique(
                &mut inventories.production_tensors,
                with_component(tensor_id, LagunaTensorComponent::Scales),
                scales,
            );
            insert_unique(
                &mut inventories.production_tensors,
                with_component(tensor_id, LagunaTensorComponent::Biases),
                biases,
            );
            inventories
                .observed_affine_profiles
                .insert((bits, group_size));
            dequantized_weight
        }
        None => {
            insert_unique(
                &mut inventories.production_tensors,
                tensor_id,
                source_weight
                    .retain()
                    .expect("native production weight should retain"),
            );
            source_weight
        }
    };
    insert_unique(
        &mut inventories.reference_tensors,
        tensor_id,
        reference_weight,
    );
}

fn insert_vector(
    inventories: &mut ReferenceTensorInventories,
    tensor_id: LagunaTensorId,
    tensor: MlxArray,
) {
    insert_unique(
        &mut inventories.production_tensors,
        tensor_id,
        tensor.retain().expect("production vector should retain"),
    );
    insert_unique(&mut inventories.reference_tensors, tensor_id, tensor);
}

fn deterministic(runtime: &MlxRuntime, shape: &[i32], dtype: MlxDtype, seed: usize) -> MlxArray {
    let element_count = shape.iter().product::<i32>() as usize;
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
        .map(|element_index| 0.9 + ((element_index as usize + seed) % 5) as f32 * 0.025)
        .collect::<Vec<_>>();
    runtime
        .array_from_f32(&values, &[width])
        .and_then(|array| runtime.astype(&array, dtype))
        .expect("normalization weight should construct")
}

fn insert_unique(
    tensors: &mut HashMap<LagunaTensorId, MlxArray>,
    tensor_id: LagunaTensorId,
    tensor: MlxArray,
) {
    assert!(tensors.insert(tensor_id, tensor).is_none());
}
