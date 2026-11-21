//! One problem report: what the person reads before sending, and exactly the
//! same fields as the JSON that would be sent.

use serde::Serialize;

use super::{CatalogModelName, MacDescription, ProblemKind};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct ProblemReport {
    pub(crate) app_version: String,
    pub(crate) mac: MacDescription,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model: Option<CatalogModelName>,
    pub(crate) problem: ProblemKind,
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
        readable_lines
    }
}

#[cfg(test)]
#[path = "problem_report_tests.rs"]
mod problem_report_tests;
