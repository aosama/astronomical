//! What the decision-model families that mark their folder with a settings
//! file (Kev, Clef) report to discovery, and how their weight bytes are
//! counted. Both keep a Qwen3.5 backbone in its own layout beside a head, so
//! their `config.json` reads as a chat model and the marker decides.

use std::fs;
use std::path::Path;

/// Family-derived metadata returned to neutral discovery orchestration.
pub(super) struct MarkedDecisionMetadata {
    pub maximum_tokens_per_question: u32,
    pub model_size_bytes: u64,
    /// The marker whose bytes the folder's revision is derived from.
    pub settings_file_name: &'static str,
}

/// The bytes of every safetensors file in `model_directory`: the backbone's
/// shards and the head.
pub(super) fn safetensors_total_bytes(model_directory: &Path) -> Option<u64> {
    let mut model_size_bytes = 0;
    for directory_entry in fs::read_dir(model_directory).ok()?.flatten() {
        if directory_entry
            .file_name()
            .to_string_lossy()
            .ends_with(".safetensors")
        {
            model_size_bytes += directory_entry.metadata().ok()?.len();
        }
    }
    Some(model_size_bytes)
}
