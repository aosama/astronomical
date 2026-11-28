//! The release speed ledger: one source-controlled file a release under
//! `registry/release_speed/`, holding every reference measurement taken of
//! that release's build. It is the one durable record of how fast
//! Astronomical serves its reference models. The release notes' speed section
//! is derived from it. The catalog does not copy the figures.
//!
//! A file a release, rather than one growing file, because a release's
//! figures are written once and then only read: two branches never edit the
//! same file, and the previous release's figures are a file to open instead
//! of a search through git history.
//!
//! Every condition a figure depends on is recorded beside it, as the engine
//! reported it while serving: the memory limit, whether the weights streamed
//! from the SSD, and the drafter in use. Two releases' figures are compared
//! only when the conditions that set the workload are equal (see
//! `comparison_key`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Run-to-run spread of one round: medians of the same build read up to 8%
/// apart on a streamed model at a tight memory limit.
const SINGLE_ROUND_SPREAD: f64 = 0.08;

pub const LEDGER_DIRECTORY: &str = "registry/release_speed";
/// The reference set lives beside the ledgers it produces, so a change to
/// what is measured is reviewed with the figures it changes.
pub const REFERENCE_SET_FILE_NAME: &str = "reference_set.json";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseSpeedLedger {
    pub astronomical_version: String,
    /// The engine's own build commit, so a figure is traceable to the exact
    /// code that served it.
    pub source_commit: String,
    pub measured_on_date: String,
    pub machine: MeasuringMachine,
    pub measurements: Vec<SpeedMeasurement>,
}

/// The Mac the figures were taken on, by chip and memory only: nothing that
/// identifies the person or their machine.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuringMachine {
    pub chip: String,
    pub memory_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedMeasurement {
    pub model_id: String,
    /// The catalog revision measured; a repackaged model is a new row, not a
    /// change in speed.
    pub model_revision: Option<String>,
    /// The memory setting the case asked for, in GB of 2^30 bytes; `None` is
    /// the Mac's own limit.
    pub maximum_mlx_memory_gb: Option<u64>,
    /// The limit the engine actually applied.
    pub mlx_memory_limit_bytes: u64,
    pub weights_placement: WeightsPlacement,
    pub drafter: DrafterInUse,
    /// The engine's effective multi-token-prediction draft count. It is a
    /// setting, reported for every model; only a model with a prediction head
    /// drafts with it.
    pub multi_token_prediction_draft_tokens: Option<u32>,
    pub task: String,
    pub temperature: f64,
    pub reply_tokens_requested: u32,
    pub figures: TaskFigures,
}

/// One task's figures over its rounds: what a person waits on before the
/// first token, and how fast the reply then comes.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskFigures {
    pub round_count: usize,
    pub prompt_tokens: u64,
    pub prefill_tokens_per_second: FigureSpread,
    pub seconds_to_first_token: FigureSpread,
    pub reply_tokens_per_second: FigureSpread,
    /// What drafting did for the task's replies; absent when none drafted,
    /// and in ledgers written before replies reported it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drafting: Option<DraftingFigures>,
    /// The most MLX memory in use during a reply, prompt reading included,
    /// in bytes; absent when the engine reported none, and in ledgers
    /// written before it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_mlx_memory_bytes: Option<FigureSpread>,
}

/// A task's drafting over its rounds: what explains a drafted reply speed,
/// and what shows a reply that stopped drafting.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DraftingFigures {
    pub rounds_a_reply: FigureSpread,
    pub kept_drafts_a_round: FigureSpread,
    /// Kept drafts as a share of the reply's tokens, between 0 and 1.
    pub drafted_share_of_reply: FigureSpread,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightsPlacement {
    InMemory,
    Streamed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DrafterInUse {
    None,
    Dflash,
    Dflash2,
}

/// The median of a figure's rounds with its lowest and highest, so a reader
/// can tell a change from run-to-run spread.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FigureSpread {
    pub median: f64,
    pub lowest: f64,
    pub highest: f64,
}

impl FigureSpread {
    /// `None` for no rounds. Figures keep two decimals: finer digits are noise
    /// and would make every re-measure a large diff.
    #[must_use]
    pub fn of_rounds(round_figures: &[f64]) -> Option<Self> {
        let mut sorted_figures = round_figures.to_vec();
        sorted_figures.sort_by(f64::total_cmp);
        let (&lowest, &highest) = (sorted_figures.first()?, sorted_figures.last()?);
        let middle_index = sorted_figures.len() / 2;
        let median = if sorted_figures.len() % 2 == 1 {
            sorted_figures[middle_index]
        } else {
            f64::midpoint(
                sorted_figures[middle_index - 1],
                sorted_figures[middle_index],
            )
        };
        Some(Self {
            median: two_decimals(median),
            lowest: two_decimals(lowest),
            highest: two_decimals(highest),
        })
    }

    /// Whether two spreads differ by more than their sittings' own spread:
    /// their ranges do not overlap, and when either is a single round (it
    /// has no range, because its case's time budget allowed one) the medians
    /// also differ by the run-to-run spread measured on this kind of work.
    #[must_use]
    pub fn is_apart_from(&self, other_spread: &Self) -> bool {
        let ranges_overlap =
            self.lowest <= other_spread.highest && self.highest >= other_spread.lowest;
        if ranges_overlap {
            return false;
        }
        let either_is_a_single_round =
            self.lowest == self.highest || other_spread.lowest == other_spread.highest;
        if !either_is_a_single_round {
            return true;
        }
        let larger_median = self.median.max(other_spread.median);
        (self.median - other_spread.median).abs() >= larger_median * SINGLE_ROUND_SPREAD
    }
}

fn two_decimals(figure: f64) -> f64 {
    (figure * 100.0).round() / 100.0
}

/// What makes two releases' figures the same workload: the model at the same
/// revision, under the same memory setting, on the same task. The drafter is
/// deliberately not part of it: a release that starts drafting a model is the
/// change a user feels, so it is compared and labelled, not hidden.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ComparisonKey {
    pub model_id: String,
    pub model_revision: Option<String>,
    pub maximum_mlx_memory_gb: Option<u64>,
    pub task: String,
}

impl SpeedMeasurement {
    #[must_use]
    pub fn comparison_key(&self) -> ComparisonKey {
        ComparisonKey {
            model_id: self.model_id.clone(),
            model_revision: self.model_revision.clone(),
            maximum_mlx_memory_gb: self.maximum_mlx_memory_gb,
            task: self.task.clone(),
        }
    }
}

#[must_use]
pub fn ledger_path(repository_root: &Path, astronomical_version: &str) -> PathBuf {
    repository_root
        .join(LEDGER_DIRECTORY)
        .join(format!("{astronomical_version}.json"))
}

pub fn read_ledger(ledger_file: &Path) -> Result<ReleaseSpeedLedger, String> {
    let ledger_text = std::fs::read_to_string(ledger_file)
        .map_err(|read_error| format!("could not read {}: {read_error}", ledger_file.display()))?;
    serde_json::from_str(&ledger_text).map_err(|parse_error| {
        format!(
            "{} is not a release speed ledger: {parse_error}",
            ledger_file.display()
        )
    })
}

/// Writes a sitting's measurements into its release's ledger, keeping cases
/// measured in an earlier sitting of the same release.
pub fn write_ledger(
    repository_root: &Path,
    fresh_ledger: ReleaseSpeedLedger,
) -> Result<PathBuf, String> {
    let ledger_file = ledger_path(repository_root, &fresh_ledger.astronomical_version);
    let existing_ledger = if ledger_file.exists() {
        Some(read_ledger(&ledger_file)?)
    } else {
        None
    };
    let ledger_to_write = merged_ledger(existing_ledger, fresh_ledger)?;
    let mut ledger_text = serde_json::to_string_pretty(&ledger_to_write)
        .map_err(|serialize_error| format!("could not write the ledger: {serialize_error}"))?;
    ledger_text.push('\n');
    if let Some(ledger_directory) = ledger_file.parent() {
        std::fs::create_dir_all(ledger_directory).map_err(|create_error| {
            format!(
                "could not create {}: {create_error}",
                ledger_directory.display()
            )
        })?;
    }
    std::fs::write(&ledger_file, ledger_text).map_err(|write_error| {
        format!("could not write {}: {write_error}", ledger_file.display())
    })?;
    Ok(ledger_file)
}

/// A fresh measurement replaces the earlier one of the same workload; every
/// other earlier measurement is kept. Sittings on different Macs or builds are
/// refused, because one file must describe one build on one Mac.
pub fn merged_ledger(
    existing_ledger: Option<ReleaseSpeedLedger>,
    fresh_ledger: ReleaseSpeedLedger,
) -> Result<ReleaseSpeedLedger, String> {
    let Some(existing_ledger) = existing_ledger else {
        return Ok(fresh_ledger);
    };
    if existing_ledger.machine != fresh_ledger.machine
        || existing_ledger.source_commit != fresh_ledger.source_commit
    {
        return Err(format!(
            "the {} ledger was measured on another Mac or build ({} on {}); move it aside to measure this one",
            existing_ledger.astronomical_version,
            existing_ledger.source_commit,
            existing_ledger.machine.chip
        ));
    }
    let fresh_keys: Vec<ComparisonKey> = fresh_ledger
        .measurements
        .iter()
        .map(SpeedMeasurement::comparison_key)
        .collect();
    let mut kept_measurements: Vec<SpeedMeasurement> = existing_ledger
        .measurements
        .into_iter()
        .filter(|earlier_measurement| !fresh_keys.contains(&earlier_measurement.comparison_key()))
        .collect();
    kept_measurements.extend(fresh_ledger.measurements);
    kept_measurements.sort_by_key(SpeedMeasurement::comparison_key);
    Ok(ReleaseSpeedLedger {
        measurements: kept_measurements,
        ..fresh_ledger
    })
}

/// The ledger of the highest release version in the directory, or `None`
/// when no release has been measured yet. The table command compares two
/// named ledgers, so only the tests ask for the newest one by itself.
#[cfg(test)]
pub(crate) fn newest_ledger(repository_root: &Path) -> Result<Option<ReleaseSpeedLedger>, String> {
    ledger_versions_newest_first(repository_root)
        .first()
        .map(|newest_version| read_ledger(&ledger_path(repository_root, newest_version)))
        .transpose()
}

/// Every measured release version, highest first; files that are not named
/// as a version (the reference set) are not ledgers.
#[must_use]
pub fn ledger_versions_newest_first(repository_root: &Path) -> Vec<String> {
    let Ok(directory_entries) = std::fs::read_dir(repository_root.join(LEDGER_DIRECTORY)) else {
        return Vec::new();
    };
    let mut numbered_versions: Vec<(Vec<u64>, String)> = Vec::new();
    for directory_entry in directory_entries.filter_map(Result::ok) {
        let Ok(file_name) = directory_entry.file_name().into_string() else {
            continue;
        };
        let Some(version_text) = file_name.strip_suffix(".json") else {
            continue;
        };
        if let Some(parts) = version_parts(version_text) {
            numbered_versions.push((parts, version_text.to_owned()));
        }
    }
    numbered_versions.sort_unstable_by(|first, second| second.cmp(first));
    numbered_versions
        .into_iter()
        .map(|(_, version_text)| version_text)
        .collect()
}

/// `0.2.134` as `[0, 2, 134]`, so `0.2.140` sorts after `0.2.99`; a name that
/// is not a version (the reference set) is not a ledger.
#[must_use]
pub fn version_parts(version_text: &str) -> Option<Vec<u64>> {
    version_text
        .split('.')
        .map(|part_text| part_text.parse::<u64>().ok())
        .collect()
}

#[cfg(test)]
#[path = "speed_ledger_tests.rs"]
pub(crate) mod speed_ledger_tests;
