//! One problem report: what the person reads before sending, and exactly the
//! same fields as the JSON that would be sent.

use serde::Serialize;

use super::{CatalogModelName, MacDescription, ProblemKind};

/// How much memory a model needed against what the Mac had free, in GB of
/// 2^30 bytes, rounded to one decimal so a report cannot fingerprint a Mac
/// by its exact free memory.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct MemoryShortfall {
    pub(crate) needed_gb: f64,
    pub(crate) available_gb: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct ProblemReport {
    pub(crate) app_version: String,
    pub(crate) mac: MacDescription,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model: Option<CatalogModelName>,
    pub(crate) problem: ProblemKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) memory: Option<MemoryShortfall>,
}

impl MemoryShortfall {
    pub(crate) fn from_bytes(needed_bytes: u64, available_bytes: u64) -> Self {
        Self {
            needed_gb: tenths_of_gb(needed_bytes),
            available_gb: tenths_of_gb(available_bytes),
        }
    }
}

impl ProblemReport {
    /// The report as the person reads it, one field per line, holding every
    /// value the JSON carries and nothing else.
    pub(crate) fn readable_lines(&self) -> Vec<String> {
        let mut readable_lines = vec![
            format!("App: Astronomical {}", self.app_version),
            format!("Mac: {}", self.mac.display_text()),
        ];
        if let Some(model) = &self.model {
            readable_lines.push(format!("Model: {}", model.display_text()));
        }
        readable_lines.push(format!("Problem: {}", self.problem.plain_description()));
        if let Some(memory) = &self.memory {
            readable_lines.push(format!(
                "Memory: needed {} GB, {} GB free",
                memory.needed_gb, memory.available_gb
            ));
        }
        readable_lines
    }
}

fn tenths_of_gb(byte_count: u64) -> f64 {
    const BYTES_PER_GB: f64 = 1_073_741_824.0;
    (byte_count as f64 / BYTES_PER_GB * 10.0).round() / 10.0
}

#[cfg(test)]
#[path = "problem_report_tests.rs"]
mod problem_report_tests;
