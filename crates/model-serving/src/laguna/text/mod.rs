//! Laguna-owned artifact normalization and compiled Poolside text protocol.

mod artifact_descriptor;
mod artifact_documents;
mod artifact_error;
mod artifact_normalizer;
mod artifact_sampler;
mod artifact_template;
mod inference_request;
mod output_parser;
mod output_parser_error;
mod preparation_error;
mod prepared_generation;
mod processor;
mod prompt_renderer;
mod request_output;
mod sampler_config;
mod template_context;
mod template_contract;
mod template_program;
mod token_decoder;
mod tokenizer;
mod tokenizer_error;
mod tool_contract;

pub use artifact_descriptor::LagunaTextArtifactDescriptor;
pub use artifact_error::LagunaTextArtifactError;
pub use artifact_normalizer::{LagunaTextArtifactNormalizer, LagunaTextArtifactSources};
pub(crate) use artifact_template::{
    MAXIMUM_TEMPLATE_BYTES, MAXIMUM_TEMPLATE_INCLUDE_DEPTH, MAXIMUM_TEMPLATE_SOURCE_COUNT,
    discover_root_template_includes, discover_template_includes,
};
pub use inference_request::{LagunaInferenceRequest, LagunaSamplingStrategy};
pub use output_parser::{LagunaOutputEvent, LagunaOutputParser};
pub use output_parser_error::LagunaOutputParserError;
pub use preparation_error::LagunaPreparationError;
pub use prepared_generation::LagunaPreparedGeneration;
pub use processor::LagunaGenerationProcessor;
pub use prompt_renderer::{LagunaPromptRenderer, LagunaPromptRendererError};
pub use request_output::{LagunaRequestOutput, LagunaRequestOutputError};
pub use sampler_config::LagunaSamplerConfig;
pub use token_decoder::LagunaTokenDecoder;
pub use tokenizer::LagunaTokenizer;
pub use tokenizer_error::LagunaTokenizerError;
