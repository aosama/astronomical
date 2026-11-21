//! What the person has chosen to send to the Astronomical developers. Nothing
//! is sent unless they choose it: problem reports start at "ask each time",
//! and the daily count stays unanswered until the person answers the one
//! question about it (issue #2413).

use serde::{Deserialize, Serialize};

use crate::AstronomicalConfig;

/// How a problem report is handled when something goes wrong.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ProblemReportChoice {
    /// Show the whole report and send it only if the person presses Send.
    #[default]
    Ask,
    /// Send each report without asking; the person chose this after seeing one.
    Always,
    /// Never offer or send a report.
    Never,
}

impl AstronomicalConfig {
    /// How problem reports are handled. It defaults to asking, so a person
    /// sees every report in full before anything leaves the Mac.
    #[must_use]
    pub fn problem_report_choice(&self) -> ProblemReportChoice {
        self.user_config_file
            .sharing
            .as_ref()
            .and_then(|sharing| sharing.problem_reports)
            .unwrap_or_default()
    }

    /// The person's answer to "May this Mac be counted?", or `None` while
    /// they have not been asked. An unanswered question counts as no, and
    /// keeping it apart from an explicit no lets the app ask exactly once.
    #[must_use]
    pub fn daily_count_answer(&self) -> Option<bool> {
        self.user_config_file
            .sharing
            .as_ref()
            .and_then(|sharing| sharing.daily_count)
    }
}
