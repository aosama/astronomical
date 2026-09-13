//! Wire-schema parsing and validation for `qwen4_exp` configurations.
//!
//! The wire document and its quantization profile are parsed in
//! `document.rs`, validated into typed geometry in `config.rs`, and every
//! failure names its rule in `error.rs`. Quantization profiles arrive
//! separately because the same document carries a default profile plus
//! per-tensor overrides whose count varies by packaging.

pub mod config;
pub mod document;
pub mod error;

pub use config::{
    Qwen4ExpConfig, Qwen4ExpHyperConnectionConfig, Qwen4ExpLayerKind,
    Qwen4ExpLinearAttentionConfig, Qwen4ExpNgramConfig, Qwen4ExpSparseAttentionConfig,
};
pub use document::{Qwen4ExpQuantizationMode, Qwen4ExpQuantizationProfile};
pub use error::Qwen4ExpConfigError;
