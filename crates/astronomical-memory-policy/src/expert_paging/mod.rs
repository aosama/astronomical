//! Pure expert-paging support arithmetic: the safetensors header parsing
//! (over `crate::safetensors` and `crate::strict_json`) and the expert
//! weight memory-cache statistics. The quantized expert manifests stay in
//! `astronomical-model-serving` — they embed model-family configuration
//! types, and the family boundary is a separate extraction step.

pub mod expert_cache_statistics;
pub mod safetensors_header;

pub use expert_cache_statistics::ExpertWeightMemoryCacheStatistics;
pub use safetensors_header::{
    SafetensorsDtype, SafetensorsHeader, SafetensorsHeaderError, TensorHeaderEntry,
    parse_safetensors_header,
};
