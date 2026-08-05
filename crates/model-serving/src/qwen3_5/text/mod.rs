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

pub use inference_request::{Qwen3_5InferenceRequest, Qwen3_5SamplingStrategy};
pub use output_parser::{Qwen3_5OutputEvent, Qwen3_5OutputParser, Qwen3_5ToolCall};
pub use output_parser_error::Qwen3_5OutputParserError;
pub use processor::{
    Qwen3_5GenerationProcessor, qwen3_5_request_enables_thinking,
    translate_qwen3_5_preparation_error, translate_request_output_error,
};
pub use prompt::{Qwen3_5PromptError, Qwen3_5PromptRenderer};
pub use request_output::{Qwen3_5RequestOutput, Qwen3_5RequestOutputError};
#[cfg(feature = "direct-mlx")]
pub use sampler::qwen3_5_apply_top_p_mask;
pub use sampler_config::{Qwen3_5SamplerConfig, discover_sampler_config};
pub use sampling_seed::resolve_sampling_seed;
pub use token_ids::{Qwen3_5TokenIds, discover_token_ids};
pub use tokenizer::{Qwen3_5TokenDecoder, Qwen3_5Tokenizer, validate_context_token_count};
pub use tokenizer_error::Qwen3_5TokenizerError;

pub(crate) use super::artifacts::ValidatedQwen3_5Artifact;
#[cfg(feature = "direct-mlx")]
pub(crate) use super::model::Qwen3_5Model;
pub(crate) use super::vision::{
    Qwen3_5ImageProcessingError, Qwen3_5ImageProcessor, Qwen3_5ProcessedImage,
};
