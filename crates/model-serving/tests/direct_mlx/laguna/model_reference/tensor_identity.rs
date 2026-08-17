//! Canonical test tensor identities and direct-affine profile resolution.

use astronomical_model_serving::{
    LagunaAttentionProjection, LagunaExpertProjection, LagunaGlobalTensorRole,
    LagunaLayerTensorRole, LagunaStorageDescriptor, LagunaTargetContract, LagunaTensorComponent,
    LagunaTensorId,
};

pub(super) fn affine_profile(
    contract: &LagunaTargetContract,
    tensor_id: LagunaTensorId,
) -> Option<(i32, i32)> {
    let LagunaStorageDescriptor::DirectAffine(storage) = contract.storage() else {
        return None;
    };
    let profile = canonical_module_name(tensor_id)
        .map(|module_name| storage.profile_for_module(&module_name))
        .unwrap_or_else(|| storage.default_profile());
    Some((profile.bits() as i32, profile.group_size() as i32))
}

fn canonical_module_name(tensor_id: LagunaTensorId) -> Option<String> {
    match tensor_id {
        LagunaTensorId::Global {
            role: LagunaGlobalTensorRole::TokenEmbedding,
            ..
        } => Some("model.embed_tokens".to_owned()),
        LagunaTensorId::Global {
            role: LagunaGlobalTensorRole::OutputHead,
            ..
        } => Some("lm_head".to_owned()),
        LagunaTensorId::Global { .. } => None,
        LagunaTensorId::Layer {
            layer_index, role, ..
        } => canonical_layer_module_name(layer_index, role),
    }
}

fn canonical_layer_module_name(layer_index: usize, role: LagunaLayerTensorRole) -> Option<String> {
    let owner_and_projection = match role {
        LagunaLayerTensorRole::Attention(projection) => (
            "self_attn",
            match projection {
                LagunaAttentionProjection::Query => "q_proj",
                LagunaAttentionProjection::Key => "k_proj",
                LagunaAttentionProjection::Value => "v_proj",
                LagunaAttentionProjection::Output => "o_proj",
                LagunaAttentionProjection::Gate => "g_proj",
            },
        ),
        LagunaLayerTensorRole::DenseFeedForward(projection) => ("mlp", projection_name(projection)),
        LagunaLayerTensorRole::SharedExpert(projection) => {
            ("mlp.shared_expert", projection_name(projection))
        }
        LagunaLayerTensorRole::RoutedExpert(projection) => {
            ("mlp.switch_mlp", projection_name(projection))
        }
        _ => return None,
    };
    Some(format!(
        "model.layers.{layer_index}.{}.{}",
        owner_and_projection.0, owner_and_projection.1
    ))
}

fn projection_name(projection: LagunaExpertProjection) -> &'static str {
    match projection {
        LagunaExpertProjection::Gate => "gate_proj",
        LagunaExpertProjection::Up => "up_proj",
        LagunaExpertProjection::Down => "down_proj",
    }
}

pub(super) fn global(role: LagunaGlobalTensorRole) -> LagunaTensorId {
    LagunaTensorId::Global {
        role,
        component: LagunaTensorComponent::Weight,
    }
}

pub(super) fn layer_id(layer_index: usize, role: LagunaLayerTensorRole) -> LagunaTensorId {
    LagunaTensorId::Layer {
        layer_index,
        role,
        component: LagunaTensorComponent::Weight,
    }
}

pub(super) fn with_component(
    tensor_id: LagunaTensorId,
    component: LagunaTensorComponent,
) -> LagunaTensorId {
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
