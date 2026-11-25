pub mod output_parser;
mod output_parser_error;
mod processor;
mod prompt;
pub mod request_output;
mod sampler_config;

mod chat_images;
mod context_token_validation;
mod template_safe_content;
mod thinking_allowance;
mod token_decoder;
mod token_ids;
mod tokenizer;
mod tokenizer_error;
mod tool_schema;

#[cfg(feature = "direct-mlx")]
pub use crate::gpu_token_sampling::apply_repetition_penalty as qwen3_5_apply_repetition_penalty;
#[cfg(feature = "direct-mlx")]
pub use crate::gpu_token_sampling::apply_top_p_mask as qwen3_5_apply_top_p_mask;
pub use crate::sampling_seed::resolve_sampling_seed;
pub use context_token_validation::validate_context_token_count;
pub use output_parser::Qwen3_5OutputEvent;
pub use output_parser::Qwen3_5OutputParser;
pub use output_parser_error::Qwen3_5OutputParserError;
pub use processor::{
    Qwen3_5GenerationProcessor, qwen3_5_request_enables_thinking,
    translate_qwen3_5_preparation_error, translate_request_output_error,
};
pub use prompt::{Qwen3_5PromptError, Qwen3_5PromptRenderer, Qwen3_5RenderedPrompt};
pub use request_output::{Qwen3_5RequestOutput, Qwen3_5RequestOutputError};
pub use sampler_config::{Qwen3_5SamplerConfig, discover_sampler_config};
pub use token_decoder::Qwen3_5TokenDecoder;
pub use token_ids::{Qwen3_5TokenIds, discover_token_ids};
pub use tokenizer::Qwen3_5Tokenizer;
pub use tokenizer_error::Qwen3_5TokenizerError;

pub use super::artifacts::ValidatedQwen3_5Artifact;
pub use super::vision::{ProcessedImage, Qwen3_5ImageProcessingError, Qwen3_5ImageProcessor};
