//! Writes measured speeds into `registry/download_catalog.json`: the median of
//! each task's rounds, rounded to whole tokens a second, with the Mac and the
//! Astronomical version they were measured on. The figures are the release
//! speed ledger's, measured at the Mac's own memory limit, and
//! `catalog_disagreements_with_ledger` proves the two still agree.
//!
//! The catalog is edited as text, entry by entry, so its layout and key order
//! stay as authored and a re-measure shows as a small diff. The catalog uses
//! four-space indentation with each entry's closing brace at eight spaces;
//! the edit relies on that and checks the result still parses.

use std::collections::BTreeMap;

use super::speed_prompts::SpeedTask;
use super::speed_rounds::whole_rate;
use crate::release_speed::speed_ledger::ReleaseSpeedLedger;

const ENTRY_CLOSING: &str = "\n        }";
const MEASURED_SPEED_OPENING: &str = ",\n            \"measured_speed\": {";
const MEASURED_SPEED_CLOSING: &str = "\n            }";
/// The order the catalog lists a speed's tasks in: both essay rates, then both
/// code rates, as the schema declares them.
const CATALOG_TASK_ORDER: [SpeedTask; 4] = [
    SpeedTask::ShortPromptEssay,
    SpeedTask::LongPromptEssay,
    SpeedTask::ShortPromptCode,
    SpeedTask::LongPromptCode,
];

/// Where the figures were measured, written beside them.
pub struct MeasurementPlace<'a> {
    pub measured_on: &'a str,
    pub astronomical_version: &'a str,
}

/// Replaces (or adds) the measured speed of the entry with this Hugging Face
/// identity and returns the new catalog text; the text must still parse.
pub fn write_measured_speed(
    catalog_text: &str,
    huggingface_id: &str,
    task_medians: &BTreeMap<&'static str, u64>,
    measurement_place: &MeasurementPlace<'_>,
) -> Result<String, String> {
    let identity_line = format!("\"huggingface_id\": \"{huggingface_id}\"");
    let entry_start = catalog_text
        .find(&identity_line)
        .ok_or_else(|| format!("the catalog has no entry for {huggingface_id}"))?;
    let entry_end = entry_start
        + catalog_text[entry_start..]
            .find(ENTRY_CLOSING)
            .ok_or_else(|| format!("the entry for {huggingface_id} has no closing brace at the catalog's indentation"))?;
    let entry_text = without_measured_speed(&catalog_text[entry_start..entry_end]);
    let updated_text = format!(
        "{}{entry_text}{}{}",
        &catalog_text[..entry_start],
        measured_speed_block(task_medians, measurement_place),
        &catalog_text[entry_end..]
    );
    serde_json::from_str::<serde_json::Value>(&updated_text)
        .map_err(|parse_error| format!("the edited catalog no longer parses: {parse_error}"))?;
    Ok(updated_text)
}

fn without_measured_speed(entry_text: &str) -> String {
    let Some(block_start) = entry_text.find(MEASURED_SPEED_OPENING) else {
        return entry_text.to_owned();
    };
    let block_end = entry_text[block_start..]
        .find(MEASURED_SPEED_CLOSING)
        .map_or(entry_text.len(), |closing_offset| {
            block_start + closing_offset + MEASURED_SPEED_CLOSING.len()
        });
    format!("{}{}", &entry_text[..block_start], &entry_text[block_end..])
}

fn measured_speed_block(
    task_medians: &BTreeMap<&'static str, u64>,
    measurement_place: &MeasurementPlace<'_>,
) -> String {
    let rate_lines: Vec<String> = CATALOG_TASK_ORDER
        .iter()
        .map(|speed_task| {
            let catalog_field = speed_task.catalog_field();
            format!(
                "                    \"{catalog_field}\": {}",
                task_medians.get(catalog_field).copied().unwrap_or_default()
            )
        })
        .collect();
    format!(
        "{MEASURED_SPEED_OPENING}\n                \"measured_on\": {},\n                \"astronomical_version\": {},\n                \"tokens_per_second\": {{\n{}\n                }}{MEASURED_SPEED_CLOSING}",
        serde_json::Value::from(measurement_place.measured_on),
        serde_json::Value::from(measurement_place.astronomical_version),
        rate_lines.join(",\n")
    )
}

/// A catalog entry's identity: its Hugging Face repository and the revision
/// the catalog pins.
#[derive(Debug, Eq, PartialEq)]
pub struct CatalogIdentity {
    pub huggingface_id: String,
    pub revision: Option<String>,
}

/// The identity of the catalog entry whose model ID (the leaf of its
/// repository) is this one.
#[must_use]
pub fn catalog_identity(catalog_text: &str, model_id: &str) -> Option<CatalogIdentity> {
    let catalog_document: serde_json::Value = serde_json::from_str(catalog_text).ok()?;
    let catalog_entry = catalog_document["entries"]
        .as_array()?
        .iter()
        .find(|catalog_entry| entry_model_id(catalog_entry) == Some(model_id))?;
    Some(CatalogIdentity {
        huggingface_id: catalog_entry["huggingface_id"].as_str()?.to_owned(),
        revision: catalog_entry["revision"].as_str().map(str::to_owned),
    })
}

fn entry_model_id(catalog_entry: &serde_json::Value) -> Option<&str> {
    catalog_entry["huggingface_id"]
        .as_str()
        .and_then(|huggingface_id| huggingface_id.rsplit('/').next())
}

/// Every catalog speed that differs from what the newest ledger measured at
/// the Mac's own memory limit, as one line each; empty when they agree. A
/// model the catalog does not list, or one measured only under a lowered
/// limit, has no catalog figure to agree with.
pub fn catalog_disagreements_with_ledger(
    catalog_text: &str,
    newest_ledger: &ReleaseSpeedLedger,
) -> Result<Vec<String>, String> {
    let catalog_document: serde_json::Value = serde_json::from_str(catalog_text)
        .map_err(|parse_error| format!("the catalog does not parse: {parse_error}"))?;
    let catalog_entries = catalog_document["entries"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut disagreements = Vec::new();
    for ledger_measurement in &newest_ledger.measurements {
        if ledger_measurement.maximum_mlx_memory_gb.is_some() {
            continue;
        }
        let Some(catalog_entry) = catalog_entries.iter().find(|catalog_entry| {
            entry_model_id(catalog_entry) == Some(ledger_measurement.model_id.as_str())
        }) else {
            continue;
        };
        let measured_speed = &catalog_entry["measured_speed"];
        let ledger_rate = whole_rate(ledger_measurement.figures.reply_tokens_per_second.median);
        let catalog_rate = measured_speed["tokens_per_second"][&ledger_measurement.task].as_u64();
        if measured_speed["astronomical_version"] != newest_ledger.astronomical_version.as_str()
            || catalog_rate != Some(ledger_rate)
        {
            disagreements.push(format!(
                "{} {}: the catalog has {catalog_rate:?} from {}, the {} ledger has {ledger_rate}",
                ledger_measurement.model_id,
                ledger_measurement.task,
                measured_speed["astronomical_version"],
                newest_ledger.astronomical_version
            ));
        }
    }
    Ok(disagreements)
}

#[cfg(test)]
#[path = "speed_catalog_tests.rs"]
mod speed_catalog_tests;
