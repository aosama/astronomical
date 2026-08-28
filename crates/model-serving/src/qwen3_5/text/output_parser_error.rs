use thiserror::Error;

/// Errors raised while turning untrusted model text into structured output.
#[derive(Debug, Error)]
pub enum Qwen3_5OutputParserError {
    /// One decoded fragment was unexpectedly large before parsing.
    #[error(
        "decoded output fragment is {actual_fragment_bytes} bytes, exceeding {maximum_fragment_bytes}"
    )]
    FragmentTooLarge {
        /// Observed fragment size.
        actual_fragment_bytes: usize,
        /// Fixed accepted maximum.
        maximum_fragment_bytes: usize,
    },
    /// The retained incomplete output exceeded its fixed memory budget.
    #[error(
        "pending Qwen3.5 output is {actual_pending_bytes} bytes, exceeding {maximum_pending_bytes}"
    )]
    PendingOutputTooLarge {
        /// Observed retained bytes.
        actual_pending_bytes: usize,
        /// Fixed accepted maximum.
        maximum_pending_bytes: usize,
    },
    /// The same tool name appeared more than once in the declared request tools.
    #[error("declared tool '{function_name}' appeared more than once")]
    DuplicateDeclaredTool {
        /// Duplicate function name.
        function_name: String,
    },
    /// A declared function schema was malformed JSON.
    #[error("declared tool schema for '{function_name}' is invalid JSON")]
    InvalidDeclaredToolSchema {
        /// Function name.
        function_name: String,
        /// JSON parser failure.
        #[source]
        source: serde_json::Error,
    },
    /// A declared function schema was not an object.
    #[error("declared tool schema for '{function_name}' must be a JSON object")]
    DeclaredToolSchemaMustBeObject {
        /// Function name.
        function_name: String,
    },
    /// A property schema was not an object.
    #[error("property '{parameter_name}' for '{function_name}' is not a JSON object")]
    InvalidToolPropertySchema {
        /// Function name.
        function_name: String,
        /// Property name.
        parameter_name: String,
    },
    /// A declared property omitted its type.
    #[error("property '{parameter_name}' for '{function_name}' omits its JSON Schema type")]
    MissingToolParameterType {
        /// Function name.
        function_name: String,
        /// Property name.
        parameter_name: String,
    },
    /// A property type union was not one supported type plus `null`.
    #[error(
        "property '{parameter_name}' for '{function_name}' has an unsupported type declaration"
    )]
    InvalidToolParameterTypeDeclaration {
        /// Function containing the unsupported property schema.
        function_name: String,
        /// Property whose type declaration was unsupported.
        parameter_name: String,
    },
    /// A declared enum was not a bounded list of strings for a string parameter.
    #[error("property '{parameter_name}' for '{function_name}' has an invalid string enum")]
    InvalidToolParameterEnum {
        /// Function containing the malformed property schema.
        function_name: String,
        /// Property whose enum declaration was malformed.
        parameter_name: String,
    },
    /// A declared tool schema exceeded the recursive validation limit.
    #[error(
        "property '{parameter_name}' for '{function_name}' exceeds the {maximum_schema_depth}-level schema depth limit"
    )]
    ToolParameterSchemaTooDeep {
        /// Function containing the too-deep schema.
        function_name: String,
        /// Property where the bound was crossed.
        parameter_name: String,
        /// Fixed accepted maximum depth.
        maximum_schema_depth: usize,
    },
    /// The schema's `required` member was malformed.
    #[error("declared required parameters for '{function_name}' are invalid")]
    InvalidRequiredToolParameters {
        /// Function name.
        function_name: String,
    },
    /// The stream ended after a partial control marker.
    #[error("Qwen3.5 output ended in a partial control marker")]
    IncompleteControlMarker,
    /// The stream ended before a reasoning block closed.
    #[error("Qwen3.5 reasoning block did not close")]
    UnclosedReasoning,
    /// A tool call omitted its required function block.
    #[error("Qwen3.5 tool call omitted a function block")]
    ToolCallMissingFunction,
    /// A tool call omitted the `>` ending its function name.
    #[error("Qwen3.5 tool call omitted the function-name terminator")]
    ToolCallMissingFunctionNameEnd,
    /// A tool call omitted its function closing marker.
    #[error("Qwen3.5 tool call omitted the function closing marker")]
    ToolCallMissingFunctionEnd,
    /// A parameter did not begin with the expected XML marker.
    #[error("Qwen3.5 tool parameter syntax is malformed")]
    MalformedToolParameter,
    /// A parameter omitted its name terminator.
    #[error("Qwen3.5 tool parameter omitted its name terminator")]
    ToolParameterMissingNameEnd,
    /// A tool call supplied a parameter absent from its declared schema.
    #[error("Qwen3.5 tool call supplied undeclared parameter '{parameter_name}'")]
    UndeclaredToolParameter {
        /// Undeclared parameter name.
        parameter_name: String,
    },
    /// A parameter omitted its closing marker.
    #[error("Qwen3.5 tool parameter omitted its closing marker")]
    ToolParameterMissingEnd,
    /// A parameter appeared twice in one call.
    #[error("Qwen3.5 tool call repeated parameter '{parameter_name}'")]
    DuplicateToolParameter {
        /// Duplicate parameter name.
        parameter_name: String,
    },
    /// A required schema parameter was absent.
    #[error("Qwen3.5 tool call omitted required parameter '{parameter_name}'")]
    MissingRequiredToolParameter {
        /// Missing parameter name.
        parameter_name: String,
    },
    /// A boolean parameter was neither `true` nor `false`.
    #[error("Qwen3.5 tool parameter is not a valid boolean")]
    InvalidBooleanToolParameter,
    /// An integer parameter could not parse as a signed JSON integer.
    #[error("Qwen3.5 tool parameter is not a valid integer")]
    InvalidIntegerToolParameter,
    /// A number parameter was non-finite or invalid.
    #[error("Qwen3.5 tool parameter is not a valid finite number")]
    InvalidNumberToolParameter,
    /// A string parameter did not match any declared enum value.
    #[error("Qwen3.5 tool parameter is outside its declared enum")]
    ToolParameterOutsideEnum,
    /// A structured value violated a nested declared schema.
    #[error("Qwen3.5 tool parameter violates its declared schema")]
    ToolParameterViolatesDeclaredSchema,
    /// An object/array parameter was not valid JSON.
    #[error("Qwen3.5 structured tool parameter is not valid JSON")]
    InvalidJsonToolParameter(#[source] serde_json::Error),
    /// A structured parameter did not match its declared array/object type.
    #[error("Qwen3.5 structured tool parameter has the wrong JSON type")]
    WrongJsonToolParameterType,
    /// The declared property type is not in the safe initial conversion subset.
    #[error("Qwen3.5 tool parameter type '{parameter_type}' is unsupported")]
    UnsupportedToolParameterType {
        /// Declared schema type.
        parameter_type: String,
    },
    /// Validated arguments unexpectedly could not serialize.
    #[error("Qwen3.5 tool arguments could not serialize")]
    SerializeToolArguments(#[source] serde_json::Error),
}

impl Qwen3_5OutputParserError {
    /// Returns a static code safe for private diagnostics without logging generated text.
    #[must_use]
    pub fn diagnostic_code(&self) -> &'static str {
        match self {
            Self::FragmentTooLarge { .. } => "fragment_too_large",
            Self::PendingOutputTooLarge { .. } => "pending_output_too_large",
            Self::DuplicateDeclaredTool { .. } => "duplicate_declared_tool",
            Self::InvalidDeclaredToolSchema { .. } => "invalid_declared_tool_schema",
            Self::DeclaredToolSchemaMustBeObject { .. } => "declared_tool_schema_must_be_object",
            Self::InvalidToolPropertySchema { .. } => "invalid_tool_property_schema",
            Self::MissingToolParameterType { .. } => "missing_tool_parameter_type",
            Self::InvalidToolParameterTypeDeclaration { .. } => {
                "invalid_tool_parameter_type_declaration"
            }
            Self::InvalidToolParameterEnum { .. } => "invalid_tool_parameter_enum",
            Self::ToolParameterSchemaTooDeep { .. } => "tool_parameter_schema_too_deep",
            Self::InvalidRequiredToolParameters { .. } => "invalid_required_tool_parameters",
            Self::IncompleteControlMarker => "incomplete_control_marker",
            Self::UnclosedReasoning => "unclosed_reasoning",
            Self::ToolCallMissingFunction => "tool_call_missing_function",
            Self::ToolCallMissingFunctionNameEnd => "tool_call_missing_function_name_end",
            Self::ToolCallMissingFunctionEnd => "tool_call_missing_function_end",
            Self::MalformedToolParameter => "malformed_tool_parameter",
            Self::ToolParameterMissingNameEnd => "tool_parameter_missing_name_end",
            Self::UndeclaredToolParameter { .. } => "undeclared_tool_parameter",
            Self::ToolParameterMissingEnd => "tool_parameter_missing_end",
            Self::DuplicateToolParameter { .. } => "duplicate_tool_parameter",
            Self::MissingRequiredToolParameter { .. } => "missing_required_tool_parameter",
            Self::InvalidBooleanToolParameter => "invalid_boolean_tool_parameter",
            Self::InvalidIntegerToolParameter => "invalid_integer_tool_parameter",
            Self::InvalidNumberToolParameter => "invalid_number_tool_parameter",
            Self::ToolParameterOutsideEnum => "tool_parameter_outside_enum",
            Self::ToolParameterViolatesDeclaredSchema => "tool_parameter_violates_declared_schema",
            Self::InvalidJsonToolParameter(_) => "invalid_json_tool_parameter",
            Self::WrongJsonToolParameterType => "wrong_json_tool_parameter_type",
            Self::UnsupportedToolParameterType { .. } => "unsupported_tool_parameter_type",
            Self::SerializeToolArguments(_) => "serialize_tool_arguments",
        }
    }
}
