//! A Kev folder's settings, read from whichever layout the folder uses:
//! `kev_decision.json` written by `convert-kev`, or the published
//! `decision.json` (see `decision_contract::kev_folder_layout`).

use std::path::Path;

use decision_contract::DecisionFailure;
use decision_contract::kev_folder_layout::KevFolderLayout;

const MAXIMUM_SETTINGS_BYTES: u64 = 64 * 1024;

/// What the engine needs from the settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KevSettings {
    /// What every logit is divided by before the softmax.
    pub temperature: f64,
    /// The width of the head's query and key projections.
    pub pointer_dimension: usize,
    /// The folder's layout, which names its head file.
    pub layout: KevFolderLayout,
}

impl KevSettings {
    /// Reads the settings of the first layout whose settings file marks the
    /// folder as Kev.
    ///
    /// # Errors
    ///
    /// Fails when no settings file marks the folder as Kev, or the one that
    /// does is too large, unreadable, or holds unusable values.
    pub fn read(model_directory: &Path) -> Result<Self, DecisionFailure> {
        for layout in KevFolderLayout::ALL {
            let Some(settings_bytes) = read_settings_file(model_directory, layout)? else {
                continue;
            };
            if layout.marks_kev(&settings_bytes) {
                return Self::from_json_bytes(layout, &settings_bytes);
            }
        }
        Err(DecisionFailure::fatal_execution(
            "the folder has no kev_decision.json or decision.json that describes a Kev model",
        ))
    }

    /// The settings `settings_bytes` holds for `layout`.
    ///
    /// # Errors
    ///
    /// Fails when the settings are not Kev's or hold unusable values.
    pub fn from_json_bytes(
        layout: KevFolderLayout,
        settings_bytes: &[u8],
    ) -> Result<Self, DecisionFailure> {
        match layout.read_settings(settings_bytes) {
            Ok(folder_settings) => Ok(Self {
                temperature: folder_settings.temperature,
                pointer_dimension: folder_settings.pointer_dimension,
                layout,
            }),
            Err(reason) => Err(unreadable_settings(layout, &reason)),
        }
    }
}

/// The bytes of `layout`'s settings file, or `None` when the folder has none.
fn read_settings_file(
    model_directory: &Path,
    layout: KevFolderLayout,
) -> Result<Option<Vec<u8>>, DecisionFailure> {
    let settings_path = model_directory.join(layout.settings_file_name());
    let Ok(settings_metadata) = std::fs::metadata(&settings_path) else {
        return Ok(None);
    };
    if settings_metadata.len() > MAXIMUM_SETTINGS_BYTES {
        return Err(unreadable_settings(layout, "it is too large"));
    }
    match std::fs::read(&settings_path) {
        Ok(settings_bytes) => Ok(Some(settings_bytes)),
        Err(_) => Err(unreadable_settings(layout, "it cannot be read")),
    }
}

fn unreadable_settings(layout: KevFolderLayout, reason: &str) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!(
        "{} cannot be used: {reason}",
        layout.settings_file_name()
    ))
}

#[cfg(test)]
#[path = "settings_tests.rs"]
mod tests;
