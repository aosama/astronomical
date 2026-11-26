//! The reference set: the models and memory limits every release is measured
//! at, read from `registry/release_speed/reference_set.json`. Keeping it in a
//! reviewed file, not on a command line, is what makes one release's figures
//! comparable with the last: the same cases, every time.

use std::path::Path;

use serde::Deserialize;

use crate::release_speed::speed_ledger::{LEDGER_DIRECTORY, REFERENCE_SET_FILE_NAME};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceSetFile {
    cases: Vec<ReferenceCase>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReferenceCase {
    pub model_id: String,
    /// The memory limit to serve the model under, in GB of 2^30 bytes; `null`
    /// is the Mac's own limit.
    pub maximum_mlx_memory_gb: Option<u64>,
    /// Tokens of the long prompts, when not the usual 10,000: a model that
    /// reads a prompt slowly (a dense 27B reads 250 to 300 tokens a second)
    /// is measured on a shorter one, so a release's measuring stays short.
    #[serde(default)]
    pub long_prompt_tokens: Option<u32>,
    /// What the case stands for, so nobody drops it without knowing.
    pub why: String,
}

pub fn load_reference_set(repository_root: &Path) -> Result<Vec<ReferenceCase>, String> {
    let reference_set_file = repository_root
        .join(LEDGER_DIRECTORY)
        .join(REFERENCE_SET_FILE_NAME);
    let reference_set_text =
        std::fs::read_to_string(&reference_set_file).map_err(|read_error| {
            format!(
                "could not read {}: {read_error}",
                reference_set_file.display()
            )
        })?;
    parse_reference_set(&reference_set_text)
}

/// Refuses an empty set and a case listed twice: either would make one
/// release's table quietly different from the next.
pub fn parse_reference_set(reference_set_text: &str) -> Result<Vec<ReferenceCase>, String> {
    let reference_set: ReferenceSetFile = serde_json::from_str(reference_set_text)
        .map_err(|parse_error| format!("the reference set is not valid: {parse_error}"))?;
    if reference_set.cases.is_empty() {
        return Err("the reference set lists no cases".to_owned());
    }
    for (case_index, reference_case) in reference_set.cases.iter().enumerate() {
        let is_repeated = reference_set.cases[..case_index]
            .iter()
            .any(|earlier_case| {
                earlier_case.model_id == reference_case.model_id
                    && earlier_case.maximum_mlx_memory_gb == reference_case.maximum_mlx_memory_gb
            });
        if is_repeated {
            return Err(format!(
                "the reference set lists {} at {:?} GB twice",
                reference_case.model_id, reference_case.maximum_mlx_memory_gb
            ));
        }
    }
    Ok(reference_set.cases)
}

#[cfg(test)]
#[path = "reference_set_tests.rs"]
mod reference_set_tests;
