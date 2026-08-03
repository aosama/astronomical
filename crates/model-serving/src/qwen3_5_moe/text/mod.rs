mod inference_request;
mod output_parser;
mod output_parser_error;
mod processor;
mod prompt;
mod request_output;
#[cfg(feature = "direct-mlx")]
pub(crate) mod sampler;
mod sampler_config;
pub(crate) mod sampling_seed;
mod token_ids;
mod tokenizer;
mod tokenizer_error;
mod tool_schema;

pub use inference_request::{Qwen3_5MoEInferenceRequest, Qwen3_5MoESamplingStrategy};
pub use output_parser::{Qwen3_5MoEOutputEvent, Qwen3_5MoEOutputParser, Qwen3_5MoEToolCall};
pub use output_parser_error::Qwen3_5MoEOutputParserError;
pub use processor::{
    Qwen3_5MoEGenerationProcessor, translate_qwen3_5_moe_preparation_error,
    translate_request_output_error,
};
pub use prompt::{Qwen3_5MoEPromptError, Qwen3_5MoEPromptRenderer};
pub use request_output::{Qwen3_5MoERequestOutput, Qwen3_5MoERequestOutputError};
#[cfg(feature = "direct-mlx")]
pub use sampler::qwen3_5_moe_apply_top_p_mask;
pub use sampler_config::{Qwen3_5MoESamplerConfig, discover_sampler_config};
pub use sampling_seed::resolve_sampling_seed;
pub use token_ids::{Qwen3_5MoETokenIds, discover_token_ids};
pub use tokenizer::{Qwen3_5MoETokenDecoder, Qwen3_5MoETokenizer, validate_context_token_count};
pub use tokenizer_error::Qwen3_5MoETokenizerError;

pub(crate) use super::artifacts::ValidatedQwen3_5MoEArtifact;
#[cfg(feature = "direct-mlx")]
pub(crate) use super::model::Qwen3_5MoEModel;
pub(crate) use super::vision::{
    Qwen3_5MoEImageProcessingError, Qwen3_5MoEImageProcessor, Qwen3_5MoEProcessedImage,
};
