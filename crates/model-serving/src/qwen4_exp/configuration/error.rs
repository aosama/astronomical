//! Typed failures for `qwen4_exp` configuration validation.
//!
//! Every error names the rule that failed, so a rejected artifact explains
//! itself instead of disappearing into a generic parse failure.

/// Why a `qwen4_exp` configuration document cannot be validated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen4ExpConfigError {
    UnsupportedArchitecture {
        provided: String,
    },
    UnsupportedModelType {
        provided: String,
    },
    UnsupportedTextModelType {
        provided: String,
    },
    UnsupportedActivationDtype {
        provided: String,
    },
    UnsupportedOutputGateType {
        provided: String,
    },
    UnsupportedMambaSsmDtype {
        provided: String,
    },
    MissingTextConfig,
    MissingField {
        field: &'static str,
    },
    LayerScheduleLengthMismatch {
        declared_layers: u32,
        schedule_length: usize,
    },
    UnknownLayerKind {
        provided: String,
    },
    LayerScheduleIntervalMismatch {
        interval: u32,
        full_attention_layers: u32,
        declared_layers: u32,
    },
    InvalidPleLayerId {
        provided: u32,
        declared_layers: u32,
    },
    NgramHeadDimMismatch {
        embedding_dim: u32,
        head_count: u32,
    },
    NgramSizeTooSmall {
        provided: u32,
    },
    UnknownQuantizationMode {
        provided: String,
    },
    PartialFieldGroup {
        group: &'static str,
    },
}

impl std::fmt::Display for Qwen4ExpConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedArchitecture { provided } => {
                write!(formatter, "unsupported architectures entry: {provided}")
            }
            Self::UnsupportedModelType { provided } => {
                write!(formatter, "unsupported model_type: {provided}")
            }
            Self::UnsupportedTextModelType { provided } => {
                write!(formatter, "unsupported text model_type: {provided}")
            }
            Self::UnsupportedActivationDtype { provided } => {
                write!(formatter, "unsupported activation dtype: {provided}")
            }
            Self::UnsupportedOutputGateType { provided } => {
                write!(formatter, "unsupported output_gate_type: {provided}")
            }
            Self::UnsupportedMambaSsmDtype { provided } => {
                write!(formatter, "unsupported mamba_ssm_dtype: {provided}")
            }
            Self::MissingTextConfig => {
                write!(formatter, "configuration has no text_config document")
            }
            Self::MissingField { field } => {
                write!(formatter, "text configuration is missing {field}")
            }
            Self::LayerScheduleLengthMismatch {
                declared_layers,
                schedule_length,
            } => write!(
                formatter,
                "layer_types holds {schedule_length} entries for {declared_layers} declared layers"
            ),
            Self::UnknownLayerKind { provided } => {
                write!(formatter, "unknown layer_types entry: {provided}")
            }
            Self::LayerScheduleIntervalMismatch {
                interval,
                full_attention_layers,
                declared_layers,
            } => write!(
                formatter,
                "full_attention_interval {interval} expects {} full-attention layers for {declared_layers} layers, found {full_attention_layers}",
                declared_layers / (*interval).max(1)
            ),
            Self::InvalidPleLayerId {
                provided,
                declared_layers,
            } => write!(
                formatter,
                "ple_layer_ids entry {provided} is outside the one-based layer range 1..={declared_layers}"
            ),
            Self::NgramHeadDimMismatch {
                embedding_dim,
                head_count,
            } => write!(
                formatter,
                "ple_embed_dim {embedding_dim} is not divisible by the {head_count} n-gram hash heads"
            ),
            Self::NgramSizeTooSmall { provided } => {
                write!(formatter, "ngram_size must be at least 2, got {provided}")
            }
            Self::UnknownQuantizationMode { provided } => {
                write!(formatter, "unknown quantization mode: {provided}")
            }
            Self::PartialFieldGroup { group } => write!(
                formatter,
                "the {group} field group is partially declared; every field in the group is required"
            ),
        }
    }
}

impl std::error::Error for Qwen4ExpConfigError {}
