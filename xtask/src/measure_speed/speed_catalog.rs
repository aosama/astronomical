//! Writes measured speeds into `registry/download_catalog.json`: the median of
//! each task's rounds, rounded to whole tokens a second, with the Mac and the
//! Astronomical version they were measured on.
//!
//! The catalog is edited as text, entry by entry, so its layout and key order
//! stay as authored and a re-measure shows as a small diff. The catalog uses
//! four-space indentation with each entry's closing brace at eight spaces;
//! the edit relies on that and checks the result still parses.

use std::collections::BTreeMap;

use super::speed_prompts::SpeedTask;

/// Rounds a task needs before its median goes in the catalog.
pub const REQUIRED_ROUND_COUNT: usize = 3;
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

/// The measured rates of one model, every round of every task.
#[derive(Debug, Default)]
pub struct ModelSpeedRounds {
    rates_by_task: BTreeMap<&'static str, Vec<f64>>,
}

impl ModelSpeedRounds {
    pub fn record(&mut self, speed_task: SpeedTask, reply_tokens_per_second: f64) {
        self.rates_by_task
            .entry(speed_task.catalog_field())
            .or_default()
            .push(reply_tokens_per_second);
    }

    /// Whole-token medians for every task, or `None` until each task has the
    /// required rounds; a partial set is never written.
    #[must_use]
    pub fn medians(&self) -> Option<BTreeMap<&'static str, u64>> {
        SpeedTask::ALL
            .iter()
            .map(|speed_task| {
                let task_rates = self.rates_by_task.get(speed_task.catalog_field())?;
                (task_rates.len() >= REQUIRED_ROUND_COUNT)
                    .then(|| (speed_task.catalog_field(), whole_median(task_rates)))
            })
            .collect()
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a rounded positive rate of tokens a second"
)]
fn whole_median(task_rates: &[f64]) -> u64 {
    let mut sorted_rates = task_rates.to_vec();
    sorted_rates.sort_by(f64::total_cmp);
    let middle_index = sorted_rates.len() / 2;
    let median_rate = if sorted_rates.len() % 2 == 1 {
        sorted_rates[middle_index]
    } else {
        f64::midpoint(sorted_rates[middle_index - 1], sorted_rates[middle_index])
    };
    median_rate.round() as u64
}

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

#[cfg(test)]
#[path = "speed_catalog_tests.rs"]
mod speed_catalog_tests;
