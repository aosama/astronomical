//! Descriptor-retaining validation of the Qwen-Image-2.1 MLX nested Diffusers tree.
//!
//! The validator opens every required file exactly once, proves the package is the reviewed
//! family (strict configs, exact physical tensor profiles, index/physical size agreement), and
//! hands the engine one owned file handle per component. Rejection messages are path-free so an
//! artifact error never leaks a local directory into a log or a REST response.

mod component_validation;
pub mod error;
mod provenance;
mod validated_artifact;
mod validator;

pub use error::QwenImage21ArtifactError;
pub use provenance::{
    QWEN_IMAGE_21_LICENSE_IDENTIFIER, QWEN_IMAGE_21_OFFICIAL_MODEL_ID,
    QWEN_IMAGE_21_PROVIDER_MODEL_ID, QwenImage21ArtifactProvenance, QwenImage21License,
};
pub use validated_artifact::{QwenImage21RetainedArtifactFiles, ValidatedQwenImage21Artifact};
pub use validator::QwenImage21ArtifactValidator;

const MODEL_INDEX_FILE_NAME: &str = "model_index.json";
const SCHEDULER_CONFIG_FILE_NAME: &str = "scheduler/scheduler_config.json";
const TEXT_ENCODER_CONFIG_FILE_NAME: &str = "text_encoder/config.json";
const TRANSFORMER_CONFIG_FILE_NAME: &str = "transformer/config.json";
const VAE_CONFIG_FILE_NAME: &str = "vae/config.json";
const TEXT_ENCODER_WEIGHTS_FILE_NAME: &str = "text_encoder/model.safetensors";
const TRANSFORMER_WEIGHTS_FILE_NAME: &str = "transformer/model.safetensors";
const VAE_WEIGHTS_FILE_NAME: &str = "vae/model.safetensors";
const TEXT_ENCODER_INDEX_FILE_NAME: &str = "text_encoder/model.safetensors.index.json";
const TRANSFORMER_INDEX_FILE_NAME: &str = "transformer/model.safetensors.index.json";
const VAE_INDEX_FILE_NAME: &str = "vae/model.safetensors.index.json";
/// Every config document the validator opens, assembled from the named paths so a reader never
/// has to cross-reference an index against the call order.
const CONFIG_FILE_NAMES: [&str; 4] = [
    SCHEDULER_CONFIG_FILE_NAME,
    TEXT_ENCODER_CONFIG_FILE_NAME,
    TRANSFORMER_CONFIG_FILE_NAME,
    VAE_CONFIG_FILE_NAME,
];
const INDEX_FILE_NAMES: [&str; 3] = [
    TEXT_ENCODER_INDEX_FILE_NAME,
    TRANSFORMER_INDEX_FILE_NAME,
    VAE_INDEX_FILE_NAME,
];
const PROCESSOR_SIDECAR_FILE_NAMES: [&str; 9] = [
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer_config.json",
    "processor/tokenizer.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
];
