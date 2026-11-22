//! One problem report: what the person reads before sending, and exactly the
//! same fields as the JSON that would be sent.

use serde::Serialize;

use super::{CatalogModelName, MacDescription, ProblemKind, ReportFact};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct ProblemReport {
    pub(crate) app_version: String,
    pub(crate) mac: MacDescription,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model: Option<CatalogModelName>,
    pub(crate) problem: ProblemKind,
    /// What explains the problem: fixed names with numbers or fixed words.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) facts: Vec<ReportFact>,
    /// What the engine did just before, oldest first, as fixed phrases with
    /// numbers ("12 s before: model loaded").
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) recent_events: Vec<String>,
    /// What the person typed about the problem, if anything. The only free
    /// text a report can hold, and it is theirs: they wrote it on the same
    /// screen that shows the rest of the report.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) note: Option<String>,
}

/// The longest note kept, in characters: room for a few sentences, short
/// enough that a pasted document cannot ride along.
pub(crate) const MAXIMUM_NOTE_CHARACTERS: usize = 500;

impl ProblemReport {
    /// Attaches the person's note, trimmed and cut to the limit; an empty
    /// note attaches nothing.
    pub(crate) fn with_note(mut self, typed_note: &str) -> Self {
        let kept_note: String = typed_note
            .trim()
            .chars()
            .take(MAXIMUM_NOTE_CHARACTERS)
            .collect();
        self.note = (!kept_note.is_empty()).then_some(kept_note);
        self
    }

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
        for fact in &self.facts {
            readable_lines.push(fact.readable_line());
        }
        if !self.recent_events.is_empty() {
            readable_lines.push("Before this:".to_owned());
            for recent_event in &self.recent_events {
                readable_lines.push(format!("  {recent_event}"));
            }
        }
        if let Some(note) = &self.note {
            readable_lines.push(format!("Your note: {note}"));
        }
        readable_lines
    }
}

#[cfg(test)]
#[path = "problem_report_tests.rs"]
mod problem_report_tests;
