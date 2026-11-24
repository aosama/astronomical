//! Failures while reading an artifact. Each message names what a person can act
//! on, because these reach the command line unchanged.

use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VisionTowerError {
    #[error(
        "{reference} is neither a local model directory nor a Hugging Face repository like organization/name or organization/name@revision"
    )]
    UnrecognizedModelReference { reference: String },
    #[error("could not build the Hugging Face HTTP client: {cause}")]
    HttpClientUnavailable { cause: String },
    #[error("the request to {url} failed: {cause}")]
    HttpRequestFailed { url: String, cause: String },
    #[error(
        "Hugging Face has no public repository {repository_id} at revision {revision}; private and gated repositories are not supported yet"
    )]
    RepositoryNotFound {
        repository_id: String,
        revision: String,
    },
    #[error("{url} answered with HTTP status {status}")]
    UnexpectedHttpStatus { url: String, status: u16 },
    #[error("{url} answered with more than the {maximum_bytes} bytes expected")]
    ReplyTooLarge { url: String, maximum_bytes: u64 },
    #[error("{url} did not honor the byte range requested: {description}")]
    RangeNotHonored { url: String, description: String },
    #[error("the Hugging Face repository listing could not be read: {description}")]
    RepositoryListingMalformed { description: String },
    #[error("could not read {path}: {cause}")]
    FileUnreadable { path: PathBuf, cause: String },
    #[error("{file} is larger than the {maximum_bytes} bytes allowed for this kind of file")]
    FileTooLarge { file: String, maximum_bytes: u64 },
    #[error("{file} is not a valid safetensors file: {description}")]
    SafetensorsMalformed { file: String, description: String },
    #[error("{file} is not valid JSON: {description}")]
    JsonMalformed { file: String, description: String },
    #[error("this artifact has more than the {maximum_count} weight files supported")]
    TooManyWeightFiles { maximum_count: usize },
    #[error(
        "the text model must be a local directory, so its weights can be copied without downloading them twice; run hf download {reference} and pass the folder it prints"
    )]
    TextModelNotLocal { reference: String },
    #[error("{path} already exists, and vision add never overwrites anything")]
    OutputAlreadyExists { path: PathBuf },
    #[error("{path} is left over from an interrupted run; remove it and try again")]
    IncompleteOutputPresent { path: PathBuf },
    #[error("this tower was not added: {reason}")]
    AdditionRefused { reason: String },
    #[error("{tensor_name} uses a tower layout that cannot be converted for MLX yet")]
    UnsupportedTowerNaming { tensor_name: String },
    #[error("the assembled model failed its final check: {description}")]
    AssembledModelInvalid { description: String },
    #[error("could not write {path}: {cause}")]
    FileUnwritable { path: PathBuf, cause: String },
}
