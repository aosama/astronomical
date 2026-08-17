//! Resident Laguna weights keyed only by canonical tensor IDs.

use std::collections::HashMap;

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};

use crate::laguna::artifacts::{
    LagunaAttentionProjection, LagunaExpertProjection, LagunaGlobalTensorRole,
    LagunaLayerTensorRole, LagunaTensorComponent, LagunaTensorId, laguna_canonical_module_name,
};
use crate::laguna::normalization::{
    LagunaFeedForwardDescriptor, LagunaGatingKind, LagunaStorageDescriptor, LagunaTargetContract,
};

use super::bound_linear::{
    LagunaBoundLinear, is_floating_weight, require_supported_affine_profile,
};
use super::error::LagunaExecutionError;
use super::router_correction_bias::bind_optional_router_correction_bias;

/// Resident weight map bound from canonical tensor IDs.
pub struct LagunaNativeWeights {
    vectors: HashMap<LagunaTensorId, MlxArray>,
    pub(super) linears: HashMap<LagunaTensorId, LagunaBoundLinear>,
    pub(super) fused_routed_gate_up: HashMap<usize, LagunaBoundLinear>,
    fused_shared_gate_up: HashMap<usize, LagunaBoundLinear>,
}

impl LagunaNativeWeights {
    /// Accepts caller-owned arrays and requires every executable projection.
    pub fn bind(
        runtime: &MlxRuntime,
        mut tensors: HashMap<LagunaTensorId, MlxArray>,
        contract: &LagunaTargetContract,
    ) -> Result<Self, LagunaExecutionError> {
        let mut vectors = HashMap::new();
        let mut linears = HashMap::new();
        bind_embedding(
            runtime,
            &mut tensors,
            &mut vectors,
            contract,
            global_id(LagunaGlobalTensorRole::TokenEmbedding),
            "token embedding weight is required",
        )?;
        bind_vector(
            &mut tensors,
            &mut vectors,
            global_id(LagunaGlobalTensorRole::FinalNormalization),
            "final normalization weight is required",
        )?;
        if !contract.model().has_tied_embeddings() {
            bind_linear(
                &mut tensors,
                &mut linears,
                contract,
                global_id(LagunaGlobalTensorRole::OutputHead),
                "untied output-head weight is required",
            )?;
        }
        let mut fused_routed_gate_up = HashMap::new();
        let mut fused_shared_gate_up = HashMap::new();
        for layer_descriptor in contract.layers() {
            let layer_index = layer_descriptor.layer_index();
            bind_vector(
                &mut tensors,
                &mut vectors,
                layer_id(layer_index, LagunaLayerTensorRole::InputNormalization),
                "input normalization weight is required",
            )?;
            bind_vector(
                &mut tensors,
                &mut vectors,
                layer_id(
                    layer_index,
                    LagunaLayerTensorRole::PostAttentionNormalization,
                ),
                "post-attention normalization weight is required",
            )?;
            bind_vector(
                &mut tensors,
                &mut vectors,
                layer_id(
                    layer_index,
                    LagunaLayerTensorRole::AttentionQueryNormalization,
                ),
                "query normalization weight is required",
            )?;
            bind_vector(
                &mut tensors,
                &mut vectors,
                layer_id(
                    layer_index,
                    LagunaLayerTensorRole::AttentionKeyNormalization,
                ),
                "key normalization weight is required",
            )?;
            for projection in [
                LagunaAttentionProjection::Query,
                LagunaAttentionProjection::Key,
                LagunaAttentionProjection::Value,
                LagunaAttentionProjection::Output,
            ] {
                bind_linear(
                    &mut tensors,
                    &mut linears,
                    contract,
                    layer_id(layer_index, LagunaLayerTensorRole::Attention(projection)),
                    "attention projection weight is required",
                )?;
            }
            if layer_descriptor.attention().gating_kind() != LagunaGatingKind::None {
                bind_linear(
                    &mut tensors,
                    &mut linears,
                    contract,
                    layer_id(
                        layer_index,
                        LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Gate),
                    ),
                    "attention gate weight is required",
                )?;
            }
            match layer_descriptor.feed_forward() {
                LagunaFeedForwardDescriptor::Dense(_) => {
                    for projection in [
                        LagunaExpertProjection::Gate,
                        LagunaExpertProjection::Up,
                        LagunaExpertProjection::Down,
                    ] {
                        bind_linear(
                            &mut tensors,
                            &mut linears,
                            contract,
                            layer_id(
                                layer_index,
                                LagunaLayerTensorRole::DenseFeedForward(projection),
                            ),
                            "dense feed-forward weight is required",
                        )?;
                    }
                }
                LagunaFeedForwardDescriptor::Moe(moe_descriptor) => {
                    bind_linear(
                        &mut tensors,
                        &mut linears,
                        contract,
                        layer_id(layer_index, LagunaLayerTensorRole::Router),
                        "router weight is required",
                    )?;
                    bind_optional_router_correction_bias(
                        &mut tensors,
                        &mut vectors,
                        layer_index,
                        moe_descriptor.expert_count(),
                    )?;
                    let routed_projections = [
                        LagunaExpertProjection::Gate,
                        LagunaExpertProjection::Up,
                        LagunaExpertProjection::Down,
                    ];
                    let present_routed_projection_count = routed_projections
                        .iter()
                        .filter(|projection| {
                            tensors.contains_key(&layer_id(
                                layer_index,
                                LagunaLayerTensorRole::RoutedExpert(**projection),
                            ))
                        })
                        .count();
                    if present_routed_projection_count == routed_projections.len() {
                        for projection in routed_projections {
                            bind_linear(
                                &mut tensors,
                                &mut linears,
                                contract,
                                layer_id(
                                    layer_index,
                                    LagunaLayerTensorRole::RoutedExpert(projection),
                                ),
                                "stacked routed-expert weight is required",
                            )?;
                        }
                    } else if present_routed_projection_count != 0 {
                        return Err(LagunaExecutionError::invalid_geometry(
                            "stacked routed-expert projections must be supplied together",
                        ));
                    }
                    if moe_descriptor.shared_expert_intermediate_size() > 0 {
                        for projection in [
                            LagunaExpertProjection::Gate,
                            LagunaExpertProjection::Up,
                            LagunaExpertProjection::Down,
                        ] {
                            bind_linear(
                                &mut tensors,
                                &mut linears,
                                contract,
                                layer_id(
                                    layer_index,
                                    LagunaLayerTensorRole::SharedExpert(projection),
                                ),
                                "shared-expert weight is required",
                            )?;
                        }
                        fuse_layer_gate_up(
                            runtime,
                            &mut linears,
                            &mut fused_shared_gate_up,
                            layer_index,
                            LagunaLayerTensorRole::SharedExpert,
                        )?;
                    }
                    let gate_id = layer_id(
                        layer_index,
                        LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Gate),
                    );
                    let up_id = layer_id(
                        layer_index,
                        LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Up),
                    );
                    let fused_gate_up = match (linears.get(&gate_id), linears.get(&up_id)) {
                        (Some(gate), Some(up)) => {
                            LagunaBoundLinear::fuse_matching_affine_output_rows(runtime, gate, up)?
                        }
                        _ => None,
                    };
                    if let Some(fused_gate_up) = fused_gate_up {
                        // MLX concatenation is lazy. Materialize the replacement,
                        // then remove split sources so resident weights own one
                        // gate/up representation rather than both.
                        fused_gate_up.materialize_storage(runtime)?;
                        linears.remove(&gate_id);
                        linears.remove(&up_id);
                        fused_routed_gate_up.insert(layer_index, fused_gate_up);
                    }
                }
            }
        }
        Ok(Self {
            vectors,
            linears,
            fused_routed_gate_up,
            fused_shared_gate_up,
        })
    }

    pub(in crate::laguna) fn global(
        &self,
        role: LagunaGlobalTensorRole,
    ) -> Result<&MlxArray, LagunaExecutionError> {
        self.vectors.get(&global_id(role)).ok_or_else(|| {
            LagunaExecutionError::missing_weight("a required global weight is missing")
        })
    }

    pub(in crate::laguna) fn global_linear(
        &self,
        role: LagunaGlobalTensorRole,
    ) -> Result<&LagunaBoundLinear, LagunaExecutionError> {
        self.linears.get(&global_id(role)).ok_or_else(|| {
            LagunaExecutionError::missing_weight("a required global projection is missing")
        })
    }

    pub(in crate::laguna) fn layer(
        &self,
        layer_index: usize,
        role: LagunaLayerTensorRole,
    ) -> Result<&MlxArray, LagunaExecutionError> {
        self.vectors
            .get(&layer_id(layer_index, role))
            .ok_or_else(|| {
                LagunaExecutionError::missing_weight("a required layer weight is missing")
            })
    }

    pub(in crate::laguna) fn linear(
        &self,
        layer_index: usize,
        role: LagunaLayerTensorRole,
    ) -> Result<&LagunaBoundLinear, LagunaExecutionError> {
        self.linears
            .get(&layer_id(layer_index, role))
            .ok_or_else(|| {
                LagunaExecutionError::missing_weight("a required layer projection is missing")
            })
    }

    pub(in crate::laguna) fn optional_layer(
        &self,
        layer_index: usize,
        role: LagunaLayerTensorRole,
    ) -> Option<&MlxArray> {
        self.vectors.get(&layer_id(layer_index, role))
    }

    pub(in crate::laguna) fn fused_routed_gate_up(
        &self,
        layer_index: usize,
    ) -> Option<&LagunaBoundLinear> {
        self.fused_routed_gate_up.get(&layer_index)
    }

    pub(in crate::laguna) fn fused_shared_gate_up(
        &self,
        layer_index: usize,
    ) -> Option<&LagunaBoundLinear> {
        self.fused_shared_gate_up.get(&layer_index)
    }
}

fn fuse_layer_gate_up(
    runtime: &MlxRuntime,
    linears: &mut HashMap<LagunaTensorId, LagunaBoundLinear>,
    fused_gate_up_by_layer: &mut HashMap<usize, LagunaBoundLinear>,
    layer_index: usize,
    role_for_projection: fn(LagunaExpertProjection) -> LagunaLayerTensorRole,
) -> Result<(), LagunaExecutionError> {
    let gate_id = layer_id(
        layer_index,
        role_for_projection(LagunaExpertProjection::Gate),
    );
    let up_id = layer_id(layer_index, role_for_projection(LagunaExpertProjection::Up));
    let fused_gate_up = match (linears.get(&gate_id), linears.get(&up_id)) {
        (Some(gate), Some(up)) => {
            LagunaBoundLinear::fuse_matching_affine_output_rows(runtime, gate, up)?
        }
        _ => None,
    };
    if let Some(fused_gate_up) = fused_gate_up {
        // Evaluate the replacement before dropping the split sources because
        // MLX concatenation retains those sources until materialization.
        fused_gate_up.materialize_storage(runtime)?;
        linears.remove(&gate_id);
        linears.remove(&up_id);
        fused_gate_up_by_layer.insert(layer_index, fused_gate_up);
    }
    Ok(())
}

fn bind_embedding(
    runtime: &MlxRuntime,
    tensors: &mut HashMap<LagunaTensorId, MlxArray>,
    vectors: &mut HashMap<LagunaTensorId, MlxArray>,
    contract: &LagunaTargetContract,
    tensor_id: LagunaTensorId,
    description: &'static str,
) -> Result<(), LagunaExecutionError> {
    let weight = tensors
        .remove(&tensor_id)
        .ok_or_else(|| LagunaExecutionError::missing_weight(description))?;
    let scales = tensors.remove(&with_component(tensor_id, LagunaTensorComponent::Scales));
    let biases = tensors.remove(&with_component(tensor_id, LagunaTensorComponent::Biases));
    let embedding_weight = match (scales, biases) {
        (None, None) => {
            if !is_floating_weight(weight.dtype()) {
                return Err(LagunaExecutionError::invalid_geometry(
                    "unquantized embeddings must use a floating-point weight",
                ));
            }
            weight
        }
        (Some(scales), Some(biases)) => {
            let (bits, group_size) = affine_profile(contract, tensor_id)?;
            runtime.dequantize_affine(&weight, &scales, &biases, group_size, bits)?
        }
        _ => {
            return Err(LagunaExecutionError::invalid_geometry(
                "embedding affine scales and biases must be supplied together",
            ));
        }
    };
    vectors.insert(tensor_id, embedding_weight);
    Ok(())
}

fn bind_vector(
    tensors: &mut HashMap<LagunaTensorId, MlxArray>,
    vectors: &mut HashMap<LagunaTensorId, MlxArray>,
    tensor_id: LagunaTensorId,
    description: &'static str,
) -> Result<(), LagunaExecutionError> {
    let weight = tensors
        .remove(&tensor_id)
        .ok_or_else(|| LagunaExecutionError::missing_weight(description))?;
    if tensors.contains_key(&with_component(tensor_id, LagunaTensorComponent::Scales))
        || tensors.contains_key(&with_component(tensor_id, LagunaTensorComponent::Biases))
    {
        return Err(LagunaExecutionError::invalid_geometry(
            "vector tensors cannot carry affine scale or bias sidecars",
        ));
    }
    vectors.insert(tensor_id, weight);
    Ok(())
}

fn bind_linear(
    tensors: &mut HashMap<LagunaTensorId, MlxArray>,
    linears: &mut HashMap<LagunaTensorId, LagunaBoundLinear>,
    contract: &LagunaTargetContract,
    tensor_id: LagunaTensorId,
    description: &'static str,
) -> Result<(), LagunaExecutionError> {
    let weight = tensors
        .remove(&tensor_id)
        .ok_or_else(|| LagunaExecutionError::missing_weight(description))?;
    let scales = tensors.remove(&with_component(tensor_id, LagunaTensorComponent::Scales));
    let biases = tensors.remove(&with_component(tensor_id, LagunaTensorComponent::Biases));
    let bound = match (scales, biases) {
        (None, None) => {
            if !is_floating_weight(weight.dtype()) {
                return Err(LagunaExecutionError::invalid_geometry(
                    "unquantized projections must use a floating-point weight",
                ));
            }
            LagunaBoundLinear::Native { weight }
        }
        (Some(scales), Some(biases)) => {
            let (bits, group_size) = affine_profile(contract, tensor_id)?;
            if weight.dtype() != MlxDtype::UInt32 {
                return Err(LagunaExecutionError::invalid_geometry(
                    "affine packed weights must use uint32 storage",
                ));
            }
            LagunaBoundLinear::Affine {
                packed_weight: weight,
                scales,
                biases,
                bits,
                group_size,
            }
        }
        _ => {
            return Err(LagunaExecutionError::invalid_geometry(
                "affine scales and biases must be supplied together",
            ));
        }
    };
    linears.insert(tensor_id, bound);
    Ok(())
}

fn affine_profile(
    contract: &LagunaTargetContract,
    tensor_id: LagunaTensorId,
) -> Result<(i32, i32), LagunaExecutionError> {
    let LagunaStorageDescriptor::DirectAffine(storage) = contract.storage() else {
        return Err(LagunaExecutionError::invalid_geometry(
            "affine sidecars require a direct affine storage descriptor",
        ));
    };
    let profile = laguna_canonical_module_name(tensor_id)
        .map(|module_name| storage.profile_for_module(&module_name))
        .unwrap_or_else(|| storage.default_profile());
    let bits = i32::try_from(profile.bits()).map_err(|_| {
        LagunaExecutionError::invalid_geometry("affine bit width exceeds the MLX integer range")
    })?;
    let group_size = i32::try_from(profile.group_size()).map_err(|_| {
        LagunaExecutionError::invalid_geometry("affine group size exceeds the MLX integer range")
    })?;
    require_supported_affine_profile(bits, group_size)?;
    Ok((bits, group_size))
}

fn global_id(role: LagunaGlobalTensorRole) -> LagunaTensorId {
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
