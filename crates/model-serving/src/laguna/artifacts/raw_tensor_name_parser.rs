use super::{
    tensor_id::{
        LagunaAttentionProjection, LagunaExpertProjection, LagunaGlobalTensorRole,
        LagunaLayerTensorRole, LagunaTensorComponent, LagunaTensorId,
    },
    tensor_name_error::LagunaTensorNameNormalizationError,
};

const LANGUAGE_MODEL_WRAPPER: &str = "language_model.";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LagunaRawTensorNamespace {
    Bare,
    LanguageModelWrapped,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum LagunaRawExpertProjection {
    Gate,
    Up,
    GateUp,
    Down,
}

impl LagunaRawExpertProjection {
    pub(super) const fn canonical_projection(self) -> LagunaExpertProjection {
        match self {
            Self::Gate | Self::GateUp => LagunaExpertProjection::Gate,
            Self::Up => LagunaExpertProjection::Up,
            Self::Down => LagunaExpertProjection::Down,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LagunaExpertSourcePackaging {
    Stacked,
    PerExpert,
}

pub(super) enum ParsedLagunaTensorName {
    Direct {
        namespace: LagunaRawTensorNamespace,
        tensor_id: LagunaTensorId,
    },
    RoutedExpert {
        namespace: LagunaRawTensorNamespace,
        layer_index: usize,
        projection: LagunaRawExpertProjection,
        component: LagunaTensorComponent,
        packaging: LagunaExpertSourcePackaging,
        expert_index: Option<usize>,
    },
}

impl ParsedLagunaTensorName {
    pub(super) const fn namespace(&self) -> LagunaRawTensorNamespace {
        match self {
            Self::Direct { namespace, .. } | Self::RoutedExpert { namespace, .. } => *namespace,
        }
    }
}

/// Explicitly parses the small Laguna namespace grammar without regex backtracking.
pub(super) struct LagunaRawTensorNameParser {
    layer_count: usize,
    expert_count: usize,
}

impl LagunaRawTensorNameParser {
    pub(super) const fn new(layer_count: usize, expert_count: usize) -> Self {
        Self {
            layer_count,
            expert_count,
        }
    }

    pub(super) fn parse(
        &self,
        raw_name: &str,
    ) -> Result<ParsedLagunaTensorName, LagunaTensorNameNormalizationError> {
        let (namespace, unwrapped_name) = detect_namespace(raw_name)?;
        let name_components = unwrapped_name.split('.').collect::<Vec<_>>();
        let first_component = name_components.first().copied().unwrap_or_default();
        if !matches!(first_component, "model" | "lm_head") {
            return Err(LagunaTensorNameNormalizationError::UnknownTensorRoot {
                tensor_name: raw_name.to_owned(),
            });
        }

        if let Some(tensor_id) = parse_global_tensor_id(&name_components) {
            return Ok(ParsedLagunaTensorName::Direct {
                namespace,
                tensor_id,
            });
        }
        let ["model", "layers", layer_index_text, layer_components @ ..] =
            name_components.as_slice()
        else {
            return unknown_name(raw_name);
        };
        let layer_index = layer_index_text
            .parse::<usize>()
            .map_err(|_| unknown_name_error(raw_name))?;
        if layer_index >= self.layer_count {
            return Err(LagunaTensorNameNormalizationError::InvalidLayerIndex {
                tensor_name: raw_name.to_owned(),
                layer_index,
                layer_count: self.layer_count,
            });
        }

        if let Some(tensor_id) = parse_direct_layer_tensor_id(layer_index, layer_components) {
            return Ok(ParsedLagunaTensorName::Direct {
                namespace,
                tensor_id,
            });
        }
        self.parse_routed_expert(namespace, raw_name, layer_index, layer_components)
    }

    fn parse_routed_expert(
        &self,
        namespace: LagunaRawTensorNamespace,
        raw_name: &str,
        layer_index: usize,
        layer_components: &[&str],
    ) -> Result<ParsedLagunaTensorName, LagunaTensorNameNormalizationError> {
        let (packaging, expert_index, projection_name, component_name) = match layer_components {
            ["mlp", "switch_mlp", projection, component]
            | ["mlp", "experts", projection, component] => (
                LagunaExpertSourcePackaging::Stacked,
                None,
                *projection,
                *component,
            ),
            ["mlp", "experts", expert_index_text, projection, component] => {
                let expert_index = expert_index_text
                    .parse::<usize>()
                    .map_err(|_| unknown_name_error(raw_name))?;
                if expert_index >= self.expert_count {
                    return Err(LagunaTensorNameNormalizationError::InvalidExpertIndex {
                        tensor_name: raw_name.to_owned(),
                        expert_index,
                        expert_count: self.expert_count,
                    });
                }
                (
                    LagunaExpertSourcePackaging::PerExpert,
                    Some(expert_index),
                    *projection,
                    *component,
                )
            }
            _ => return unknown_name(raw_name),
        };
        let projection = parse_raw_expert_projection(projection_name)
            .ok_or_else(|| unknown_name_error(raw_name))?;
        let component =
            parse_component(component_name).ok_or_else(|| unknown_name_error(raw_name))?;
        Ok(ParsedLagunaTensorName::RoutedExpert {
            namespace,
            layer_index,
            projection,
            component,
            packaging,
            expert_index,
        })
    }
}

fn detect_namespace(
    raw_name: &str,
) -> Result<(LagunaRawTensorNamespace, &str), LagunaTensorNameNormalizationError> {
    let Some(unwrapped_name) = raw_name.strip_prefix(LANGUAGE_MODEL_WRAPPER) else {
        return Ok((LagunaRawTensorNamespace::Bare, raw_name));
    };
    if unwrapped_name.starts_with(LANGUAGE_MODEL_WRAPPER) {
        return Err(
            LagunaTensorNameNormalizationError::RepeatedLanguageModelWrapper {
                tensor_name: raw_name.to_owned(),
            },
        );
    }
    Ok((
        LagunaRawTensorNamespace::LanguageModelWrapped,
        unwrapped_name,
    ))
}

fn parse_global_tensor_id(name_components: &[&str]) -> Option<LagunaTensorId> {
    let (role, component) = match name_components {
        ["model", "embed_tokens", component_name] => (
            LagunaGlobalTensorRole::TokenEmbedding,
            parse_component(component_name)?,
        ),
        ["model", "norm", "weight"] => (
            LagunaGlobalTensorRole::FinalNormalization,
            LagunaTensorComponent::Weight,
        ),
        ["lm_head", component_name] => (
            LagunaGlobalTensorRole::OutputHead,
            parse_component(component_name)?,
        ),
        _ => return None,
    };
    Some(LagunaTensorId::Global { role, component })
}

fn parse_direct_layer_tensor_id(
    layer_index: usize,
    layer_components: &[&str],
) -> Option<LagunaTensorId> {
    let (role, component) = match layer_components {
        ["self_attn", "k_scale"] => (
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Key),
            LagunaTensorComponent::AttentionKeyScaleMetadata,
        ),
        ["self_attn", "v_scale"] => (
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Value),
            LagunaTensorComponent::AttentionValueScaleMetadata,
        ),
        ["input_layernorm", "weight"] => (
            LagunaLayerTensorRole::InputNormalization,
            LagunaTensorComponent::Weight,
        ),
        ["post_attention_layernorm", "weight"] => (
            LagunaLayerTensorRole::PostAttentionNormalization,
            LagunaTensorComponent::Weight,
        ),
        ["self_attn", "q_norm", "weight"] => (
            LagunaLayerTensorRole::AttentionQueryNormalization,
            LagunaTensorComponent::Weight,
        ),
        ["self_attn", "k_norm", "weight"] => (
            LagunaLayerTensorRole::AttentionKeyNormalization,
            LagunaTensorComponent::Weight,
        ),
        ["self_attn", projection_name, component_name] => (
            LagunaLayerTensorRole::Attention(parse_attention_projection(projection_name)?),
            parse_component(component_name)?,
        ),
        ["mlp", "gate", "e_score_correction_bias"]
        | ["mlp", "e_score_correction_bias"]
        | ["mlp", "experts", "e_score_correction_bias"]
        | ["mlp", "switch_mlp", "e_score_correction_bias"] => (
            LagunaLayerTensorRole::RouterCorrectionBias,
            LagunaTensorComponent::Weight,
        ),
        ["mlp", "gate", component_name] => (
            LagunaLayerTensorRole::Router,
            parse_component(component_name)?,
        ),
        ["mlp", "gate", "proj", component_name] => (
            LagunaLayerTensorRole::Router,
            parse_component(component_name)?,
        ),
        ["mlp", "shared_expert", projection_name, component_name] => (
            LagunaLayerTensorRole::SharedExpert(parse_expert_projection(projection_name)?),
            parse_component(component_name)?,
        ),
        ["mlp", "shared_expert_gate", component_name] => (
            LagunaLayerTensorRole::SharedExpertGate,
            parse_component(component_name)?,
        ),
        ["mlp", projection_name, component_name] => (
            LagunaLayerTensorRole::DenseFeedForward(parse_expert_projection(projection_name)?),
            parse_component(component_name)?,
        ),
        _ => return None,
    };
    Some(LagunaTensorId::Layer {
        layer_index,
        role,
        component,
    })
}

fn parse_component(component_name: &str) -> Option<LagunaTensorComponent> {
    match component_name {
        "weight" | "weight_packed" => Some(LagunaTensorComponent::Weight),
        "scales" | "weight_scale" => Some(LagunaTensorComponent::Scales),
        "biases" => Some(LagunaTensorComponent::Biases),
        "weight_global_scale" => Some(LagunaTensorComponent::WeightGlobalScale),
        "input_global_scale" => Some(LagunaTensorComponent::InputGlobalScale),
        "weight_shape" => Some(LagunaTensorComponent::LogicalShape),
        "weight_zero_point" => Some(LagunaTensorComponent::ZeroPoint),
        _ => None,
    }
}

fn parse_attention_projection(projection_name: &str) -> Option<LagunaAttentionProjection> {
    match projection_name {
        "q_proj" => Some(LagunaAttentionProjection::Query),
        "k_proj" => Some(LagunaAttentionProjection::Key),
        "v_proj" => Some(LagunaAttentionProjection::Value),
        "o_proj" => Some(LagunaAttentionProjection::Output),
        "g_proj" => Some(LagunaAttentionProjection::Gate),
        _ => None,
    }
}

fn parse_expert_projection(projection_name: &str) -> Option<LagunaExpertProjection> {
    match projection_name {
        "gate_proj" => Some(LagunaExpertProjection::Gate),
        "up_proj" => Some(LagunaExpertProjection::Up),
        "down_proj" => Some(LagunaExpertProjection::Down),
        _ => None,
    }
}

fn parse_raw_expert_projection(projection_name: &str) -> Option<LagunaRawExpertProjection> {
    match projection_name {
        "gate_proj" => Some(LagunaRawExpertProjection::Gate),
        "up_proj" => Some(LagunaRawExpertProjection::Up),
        "gate_up_proj" => Some(LagunaRawExpertProjection::GateUp),
        "down_proj" => Some(LagunaRawExpertProjection::Down),
        _ => None,
    }
}

fn unknown_name<T>(raw_name: &str) -> Result<T, LagunaTensorNameNormalizationError> {
    Err(unknown_name_error(raw_name))
}

fn unknown_name_error(raw_name: &str) -> LagunaTensorNameNormalizationError {
    LagunaTensorNameNormalizationError::UnknownTensorName {
        tensor_name: raw_name.to_owned(),
    }
}
