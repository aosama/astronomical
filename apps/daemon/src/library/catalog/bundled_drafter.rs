//! Which draft model, if any, a model on this Mac carries in its own
//! `drafter` folder, so the Models page can say so before the model loads.
//!
//! The engine decides for real when it loads the model (it also checks that
//! the drafter pairs with the model and that the GPU can run it, and status
//! reports that). This reads only the folder's small `config.json`, with the
//! same two signals the engine reads first: a `dflash_config` and the
//! architecture name, so the label never claims a drafter the engine would
//! not even consider.

use std::path::Path;

use ipc_protocol::DraftModelGeneration;

const DRAFTER_FOLDER_NAME: &str = "drafter";
/// A drafter's `config.json` is a few kilobytes; anything larger is not one.
const MAXIMUM_DRAFTER_CONFIGURATION_BYTES: u64 = 64 * 1024;

/// The generation of the DFlash draft model in a model's own `drafter`
/// folder, or `None` when it has none.
#[must_use]
pub(crate) fn bundled_drafter_generation(model_directory: &Path) -> Option<DraftModelGeneration> {
    let configuration_path = model_directory
        .join(DRAFTER_FOLDER_NAME)
        .join("config.json");
    let configuration_bytes = std::fs::metadata(&configuration_path).ok()?.len();
    if configuration_bytes > MAXIMUM_DRAFTER_CONFIGURATION_BYTES {
        return None;
    }
    let configuration_text = std::fs::read_to_string(&configuration_path).ok()?;
    drafter_generation_of_configuration(&configuration_text)
}

/// The generation a drafter `config.json` declares, or `None` when it is not
/// a DFlash drafter.
#[must_use]
pub(crate) fn drafter_generation_of_configuration(
    configuration_text: &str,
) -> Option<DraftModelGeneration> {
    let configuration_document: serde_json::Value =
        serde_json::from_str(configuration_text).ok()?;
    configuration_document.get("dflash_config")?;
    let architectures = configuration_document.get("architectures")?.as_array()?;
    let mut drafter_generation = None;
    for architecture in architectures {
        match architecture.as_str() {
            Some("DFlash2DraftModel") => return Some(DraftModelGeneration::Dflash2),
            Some("DFlashDraftModel") => drafter_generation = Some(DraftModelGeneration::Dflash),
            _ => {}
        }
    }
    drafter_generation
}

#[cfg(test)]
#[path = "bundled_drafter_tests.rs"]
mod tests;
