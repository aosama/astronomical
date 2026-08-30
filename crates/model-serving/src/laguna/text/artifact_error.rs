use thiserror::Error;

/// A bounded failure while validating the Laguna text-side artifact contract.
#[derive(Debug, Error)]
pub enum LagunaTextArtifactError {
    #[error("Laguna {document_name} contains {actual_bytes} bytes, exceeding {maximum_bytes}")]
    DocumentTooLarge {
        document_name: &'static str,
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    #[error("Laguna {document_name} is not valid JSON")]
    MalformedJson {
        document_name: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error("Laguna {document_name} contains a duplicate JSON object field")]
    DuplicateJsonField { document_name: &'static str },
    #[error("Laguna {document_name} root must be a JSON object")]
    ExpectedJsonObject { document_name: &'static str },
    #[error("Laguna field '{field_name}' is missing or has an unsupported type")]
    InvalidField { field_name: String },
    #[error("Laguna field '{field_name}' has unsupported numeric value")]
    InvalidNumericField { field_name: String },
    #[error("Laguna text config disagrees with the canonical model field '{field_name}'")]
    ModelContractMismatch { field_name: &'static str },
    #[error("Laguna template include '{include_name}' was not supplied by the artifact")]
    MissingTemplateInclude { include_name: String },
    #[error("Laguna template include cycle was detected at '{include_name}'")]
    TemplateIncludeCycle { include_name: String },
    #[error("Laguna template include '{include_name}' exceeds the supported depth")]
    TemplateIncludeDepthExceeded { include_name: String },
    #[error("Laguna template include '{include_name}' escapes the artifact root")]
    TemplateIncludeTraversal { include_name: String },
    #[error("Laguna artifact has ambiguous inline and standalone template sources")]
    AmbiguousTemplateSource { source_count: usize },
    #[error(
        "Laguna artifact supplies too many template files: {actual_count} exceeds {maximum_count}"
    )]
    TooManyTemplateSources {
        actual_count: usize,
        maximum_count: usize,
    },
    #[error("Laguna template bytes are not valid UTF-8")]
    TemplateNotUtf8(#[source] std::str::Utf8Error),
    #[error("Laguna template contract is malformed: {description}")]
    MalformedTemplateContract { description: &'static str },
    #[error("Laguna template contract is unsupported: {description}")]
    UnsupportedTemplateContract { description: &'static str },
    #[error("Laguna template could not be compiled")]
    TemplateCompilation(#[source] minijinja::Error),
    #[error("Laguna template semantic probe could not be rendered")]
    TemplateProbeRendering(#[source] minijinja::Error),
    #[error("Laguna generation config is missing parser ID '{field_name}'")]
    MissingParserId { field_name: String },
    #[error("Laguna parser ID '{parser_id}' at '{field_name}' is unsupported")]
    UnsupportedParserId {
        field_name: String,
        parser_id: String,
    },
    #[error(
        "Laguna configured token {configured_token_id} does not match tokenizer token '{token_content}'"
    )]
    SpecialTokenMismatch {
        configured_token_id: u32,
        token_content: String,
        tokenizer_token_id: Option<u32>,
    },
    #[error("Laguna tokenizer token {token_id} has conflicting identities")]
    DuplicateTokenIdentity { token_id: u32 },
    #[error("Laguna tokenizer vocabulary could not be loaded")]
    LoadTokenizer {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
