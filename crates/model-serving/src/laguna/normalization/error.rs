use thiserror::Error;

/// A bounded parsing, ambiguity, support, or safety failure during Laguna normalization.
#[derive(Debug, Error)]
pub enum LagunaNormalizationError {
    /// The caller supplied more configuration bytes than the bounded parser accepts.
    #[error(
        "Laguna config contains {actual_bytes} bytes, exceeding the {maximum_bytes}-byte limit"
    )]
    ConfigTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    /// The bounded bytes are not one valid JSON document.
    #[error("failed to decode Laguna config JSON")]
    MalformedJson(#[source] serde_json::Error),
    /// A JSON object repeated a field, making the selected meaning order-dependent.
    #[error("Laguna config contains a duplicate JSON object field")]
    DuplicateConfigField,
    /// The root or nested configuration envelope is not a JSON object.
    #[error("Laguna config field '{field_name}' must be a JSON object")]
    ExpectedObject { field_name: String },
    /// Root and nested envelopes declare different meanings for one field.
    #[error("root and text_config contain conflicting Laguna field '{field_name}'")]
    ConflictingEnvelopeField { field_name: String },
    /// A required canonical field was not declared by either envelope.
    #[error("Laguna config is missing required field '{field_name}'")]
    MissingRequiredField { field_name: String },
    /// A number is absent, out of range, non-integral, or violates geometry constraints.
    #[error("invalid Laguna numeric field '{field_name}': {description}")]
    InvalidNumericValue {
        field_name: String,
        description: &'static str,
    },
    /// A string, boolean, or enum value has no implemented Laguna meaning.
    #[error("unsupported Laguna value at '{field_name}': '{actual_value}'")]
    UnsupportedValue {
        field_name: String,
        actual_value: String,
    },
    /// A per-layer declaration does not align with the canonical layer count.
    #[error("Laguna field '{field_name}' has {actual_count} entries, expected {expected_count}")]
    LayerArrayLengthMismatch {
        field_name: String,
        actual_count: usize,
        expected_count: usize,
    },
    /// A layer's query heads cannot be grouped over its key/value heads.
    #[error(
        "Laguna layer {layer_index} has {query_head_count} query heads, which is not divisible by {key_value_head_count} key/value heads"
    )]
    InvalidHeadDivisibility {
        layer_index: usize,
        query_head_count: u32,
        key_value_head_count: u32,
    },
    /// Boolean `true` does not identify per-head versus per-element gating without tensors.
    #[error("Laguna boolean gating=true is ambiguous without a validated gate projection shape")]
    AmbiguousGatingBoolean,
    /// A rotary parameter cannot produce safe executable geometry.
    #[error("invalid Laguna RoPE field '{field_name}': {description}")]
    InvalidRopeValue {
        field_name: String,
        description: &'static str,
    },
    /// More routed experts were requested than the model declares.
    #[error("Laguna top-K {experts_per_token} exceeds expert count {expert_count}")]
    TopKExceedsExpertCount {
        experts_per_token: u32,
        expert_count: u32,
    },
    /// Direct and alternate quantization copies do not describe one canonical layout.
    #[error("Laguna quantization documents have conflicting canonical semantics")]
    ConflictingQuantizationDocuments,
    /// Two raw module names collapse to one canonical module with different profiles.
    #[error("Laguna quantization overrides conflict for canonical module '{module_name}'")]
    ConflictingModuleOverride { module_name: String },
    /// An affine bit width, group size, mode, or module key is unsupported.
    #[error("unsupported Laguna affine value at '{location}': {description} '{actual_value}'")]
    UnsupportedQuantizationValue {
        location: String,
        description: &'static str,
        actual_value: String,
    },
    /// Config names a compressed representation that requires tensor inventory classification.
    #[error(
        "Laguna storage encoding '{encoding}' requires artifact tensor-inventory normalization"
    )]
    UnsupportedStorageEncoding { encoding: String },
}
