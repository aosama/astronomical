//! Where an instance keeps compiled Neural Engine programs between loads.
//!
//! The folder sits beside the prompt cache and never inside it: the prompt cache
//! measures and trims everything under its own root against the user's size
//! limit, and compiled programs are neither prompt state nor the user's to budget
//! there. A worker is told only the prompt cache's root, so this derives the
//! sibling from that root and keeps the layout rule in one crate.

use std::path::{Path, PathBuf};

const NEURAL_ENGINE_PROGRAM_CACHE_DIRECTORY_NAME: &str = "neural-engine-programs";

/// One model revision's folder of compiled programs, or `None` when the prompt
/// cache root has no parent to hold a sibling, which leaves programs uncached.
#[must_use]
pub fn neural_engine_program_cache_directory_for_model(
    global_prompt_cache_root_directory: &Path,
    model_id: &str,
    model_revision: &str,
) -> Option<PathBuf> {
    let instance_state_directory = global_prompt_cache_root_directory.parent()?;
    if instance_state_directory.as_os_str().is_empty() {
        return None;
    }
    Some(
        instance_state_directory
            .join(NEURAL_ENGINE_PROGRAM_CACHE_DIRECTORY_NAME)
            .join(model_id)
            .join(model_revision),
    )
}

#[cfg(test)]
#[path = "neural_engine_program_cache_tests.rs"]
mod tests;
