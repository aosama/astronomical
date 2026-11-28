//! Shallow discovery rules for Kev decision-model folders.
//!
//! A Kev folder is a Qwen3.5 backbone with Kev's adapter merged in, kept in
//! a Qwen3.5 layout, so its `config.json` reads as a Qwen3.5 model. What
//! marks it as a decision model instead is its settings file beside the
//! pointer head, in one of two layouts (`decision_contract::kev_folder_layout`):
//! `kev_decision.json` and `kev_head.safetensors` from `cargo xtask
//! convert-kev`, or `decision.json` and `head.safetensors` in the published
//! builds the download catalog installs. Discovery therefore checks for
//! those markers before it reads the config's model type.

use std::fs;
use std::path::Path;

use decision_contract::kev_folder_layout::KevFolderLayout;
use model_discovery::ModelFamily;

use model_discovery::bounded_artifact_file;
use model_discovery::marked_decision_folder::{MarkedDecisionMetadata, safetensors_total_bytes};

const MAXIMUM_SETTINGS_BYTES: u64 = 64 * 1024;
/// The longest row Kev serves: a state of up to 65,536 tokens, its state
/// token included, and 8,192 more for the question (the reference's
/// `SERVE_MAX_STATE` and `SERVE_MAX_BRANCH`).
const KEV_MAXIMUM_TOKENS_PER_QUESTION: u32 = 65_536 + 8_192;

/// Whether `model_directory` is marked as a Kev decision model.
pub fn classifies_directory(model_directory: &Path) -> bool {
    marked_layout(model_directory).is_some()
}

/// The first layout whose settings file marks the folder as Kev.
fn marked_layout(model_directory: &Path) -> Option<KevFolderLayout> {
    for layout in KevFolderLayout::ALL {
        let Ok(settings_bytes) = bounded_artifact_file::read_bounded_nonempty_file(
            &model_directory.join(layout.settings_file_name()),
            MAXIMUM_SETTINGS_BYTES,
        ) else {
            continue;
        };
        if layout.marks_kev(&settings_bytes) {
            return Some(layout);
        }
    }
    None
}

/// Validates shallow Kev completeness and derives public discovery metadata.
pub(super) fn discover_model_metadata(model_directory: &Path) -> Option<MarkedDecisionMetadata> {
    let layout = marked_layout(model_directory)?;
    let model_size_bytes = complete_folder_size_bytes(model_directory, layout)?;
    Some(MarkedDecisionMetadata {
        maximum_tokens_per_question: KEV_MAXIMUM_TOKENS_PER_QUESTION,
        model_size_bytes,
        settings_file_name: layout.settings_file_name(),
    })
}

/// The folder's safetensors bytes, when it holds `layout`'s head, the
/// tokenizer, the config and backbone weights beside the head.
fn complete_folder_size_bytes(model_directory: &Path, layout: KevFolderLayout) -> Option<u64> {
    for required_file_name in [layout.head_file_name(), "tokenizer.json", "config.json"] {
        if !model_directory.join(required_file_name).is_file() {
            return None;
        }
    }
    let model_size_bytes = safetensors_total_bytes(model_directory)?;
    // The head alone is a few megabytes; a folder without backbone weights
    // is an incomplete download, not a model.
    let head_bytes = fs::metadata(model_directory.join(layout.head_file_name()))
        .ok()?
        .len();
    if model_size_bytes <= head_bytes {
        return None;
    }
    Some(model_size_bytes)
}

/// The Kev decision family: a Qwen3.5 backbone with a pointer head over the
/// options, marked by `kev_decision.json` or a Kev `decision.json`.
pub const KEV_MODEL_FAMILY: ModelFamily = ModelFamily::named("kev");
