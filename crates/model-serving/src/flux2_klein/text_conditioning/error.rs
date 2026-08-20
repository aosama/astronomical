//! Typed failures for retained-descriptor tokenization and native MLX execution.

use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum Flux2KleinTextConditioningError {
    #[error("the FLUX.2 Klein text prompt batch must contain at least one prompt")]
    EmptyPromptBatch,
    #[error("the retained FLUX.2 Klein tokenizer descriptor is missing")]
    MissingTokenizerDescriptor,
    #[error("the retained FLUX.2 Klein tokenizer descriptor is unavailable")]
    TokenizerDescriptorIo(#[source] std::io::Error),
    #[error("the retained FLUX.2 Klein tokenizer descriptor exceeds its bounded size")]
    TokenizerDescriptorTooLarge,
    #[error("the retained FLUX.2 Klein tokenizer JSON is invalid")]
    TokenizerLoad {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("the FLUX.2 Klein prompt could not be tokenized")]
    PromptTokenization {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("the FLUX.2 Klein text batch geometry exceeds native array limits")]
    BatchGeometryOverflow,
    #[cfg(feature = "direct-mlx")]
    #[error("the FLUX.2 Klein text encoder is missing tensor '{tensor_name}'")]
    MissingTensor { tensor_name: String },
    #[cfg(feature = "direct-mlx")]
    #[error("the FLUX.2 Klein text encoder tensor '{tensor_name}' is incompatible: {description}")]
    InvalidTensor {
        tensor_name: String,
        description: &'static str,
    },
    #[cfg(feature = "direct-mlx")]
    #[error("the FLUX.2 Klein text encoder has already transferred its weights")]
    WeightsUnavailable,
    #[cfg(feature = "direct-mlx")]
    #[error("the FLUX.2 Klein text encoder layer group must contain at least one layer")]
    EmptyLayerGroup,
    #[cfg(feature = "direct-mlx")]
    #[error("the retained FLUX.2 Klein text weight descriptor could not be cloned")]
    WeightDescriptorIo(#[source] std::io::Error),
    #[cfg(feature = "direct-mlx")]
    #[error("native MLX text conditioning failed")]
    Mlx(#[from] astronomical_runtime_integration::MlxRuntimeError),
}
