use thiserror::Error;

/// Bounded failures raised while parsing untrusted Poolside model output.
#[derive(Debug, Error)]
pub enum LagunaOutputParserError {
    #[error("decoded Laguna output fragment exceeds the {maximum_bytes}-byte limit")]
    FragmentTooLarge { maximum_bytes: usize },
    #[error("pending Laguna output exceeds the {maximum_bytes}-byte limit")]
    PendingOutputTooLarge { maximum_bytes: usize },
    #[error("declared Laguna tool '{function_name}' appears more than once")]
    DuplicateDeclaredTool { function_name: String },
    #[error("declared Laguna tool schema for '{function_name}' is invalid JSON")]
    InvalidDeclaredToolSchema {
        function_name: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("declared Laguna tool schema for '{function_name}' contains a duplicate field")]
    DuplicateDeclaredToolSchemaField { function_name: String },
    #[error("declared Laguna tool schema for '{function_name}' must be an object")]
    DeclaredToolSchemaMustBeObject { function_name: String },
    #[error("declared Laguna tool schema for '{function_name}' exceeds the bounded size")]
    DeclaredToolSchemaTooLarge { function_name: String },
    #[error("declared Laguna property '{argument_name}' for '{function_name}' is invalid")]
    InvalidDeclaredToolProperty {
        function_name: String,
        argument_name: String,
    },
    #[error("declared Laguna property '{argument_name}' uses an unsupported structured type")]
    UnsupportedDeclaredToolArgumentType { argument_name: String },
    #[error("declared Laguna required arguments for '{function_name}' are invalid")]
    InvalidRequiredToolArguments { function_name: String },
    #[error("Laguna tool call repeated argument '{argument_name}'")]
    DuplicateToolArgument { argument_name: String },
    #[error("Laguna function '{function_name}' omitted required argument '{argument_name}'")]
    MissingRequiredToolArgument {
        function_name: String,
        argument_name: String,
    },
    #[error("Laguna output ended with an incomplete tool call")]
    IncompleteToolCall,
    #[error("Laguna tool call contains a nested argument marker")]
    NestedToolArgumentMarker,
    #[error("Laguna tool arguments exceed the {maximum_bytes}-byte limit")]
    ToolArgumentsTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    #[error("Laguna tool-call marker syntax is malformed")]
    MalformedToolCall,
    #[error("Laguna tool argument value does not match its declared scalar type")]
    InvalidToolArgumentValue,
    #[error("Laguna tool argument arrays and objects are unsupported")]
    StructuredToolArgumentUnsupported,
    #[error("Laguna tool arguments could not be serialized")]
    SerializeToolArguments(#[source] serde_json::Error),
    #[error("Laguna output ended in a partial control marker")]
    IncompleteControlMarker,
    #[error("Laguna output contains too many tool calls")]
    TooManyToolCalls,
}

impl LagunaOutputParserError {
    /// Returns a stable diagnostic code without model-generated payload content.
    #[must_use]
    pub const fn diagnostic_code(&self) -> &'static str {
        match self {
            Self::FragmentTooLarge { .. } => "fragment_too_large",
            Self::PendingOutputTooLarge { .. } => "pending_output_too_large",
            Self::DuplicateDeclaredTool { .. } => "duplicate_declared_tool",
            Self::InvalidDeclaredToolSchema { .. } => "invalid_declared_tool_schema",
            Self::DuplicateDeclaredToolSchemaField { .. } => "duplicate_declared_tool_schema_field",
            Self::DeclaredToolSchemaMustBeObject { .. } => "declared_tool_schema_must_be_object",
            Self::DeclaredToolSchemaTooLarge { .. } => "declared_tool_schema_too_large",
            Self::InvalidDeclaredToolProperty { .. } => "invalid_declared_tool_property",
            Self::UnsupportedDeclaredToolArgumentType { .. } => {
                "unsupported_declared_tool_argument_type"
            }
            Self::InvalidRequiredToolArguments { .. } => "invalid_required_tool_arguments",
            Self::DuplicateToolArgument { .. } => "duplicate_tool_argument",
            Self::MissingRequiredToolArgument { .. } => "missing_required_tool_argument",
            Self::IncompleteToolCall => "incomplete_tool_call",
            Self::NestedToolArgumentMarker => "nested_tool_argument_marker",
            Self::ToolArgumentsTooLarge { .. } => "tool_arguments_too_large",
            Self::MalformedToolCall => "malformed_tool_call",
            Self::InvalidToolArgumentValue => "invalid_tool_argument_value",
            Self::StructuredToolArgumentUnsupported => "structured_tool_argument_unsupported",
            Self::SerializeToolArguments(_) => "serialize_tool_arguments",
            Self::IncompleteControlMarker => "incomplete_control_marker",
            Self::TooManyToolCalls => "too_many_tool_calls",
        }
    }

    /// Resource bounds stay fatal after `</tool_call>`. Every other closed-envelope failure is forwarded.
    #[must_use]
    pub(crate) const fn closed_envelope_must_remain_fatal(&self) -> bool {
        matches!(
            self,
            Self::FragmentTooLarge { .. }
                | Self::PendingOutputTooLarge { .. }
                | Self::ToolArgumentsTooLarge { .. }
                | Self::TooManyToolCalls
        )
    }
}
