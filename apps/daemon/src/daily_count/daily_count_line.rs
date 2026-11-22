//! The one line a counted Mac sends, and the same line as the person reads
//! it before agreeing.

use serde_json::json;

use crate::problem_reports::{MacDescription, sentry_envelope_for_event};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DailyCountLine {
    pub(crate) app_version: String,
    pub(crate) mac: MacDescription,
}

impl DailyCountLine {
    /// "Astronomical 0.2.126 · macOS 27.2 · Apple M5 Pro · 48 GB": every
    /// value that is sent, in the words shown on the card and in Settings.
    pub(crate) fn readable_line(&self) -> String {
        format!(
            "Astronomical {} · macOS {} · {} · {} GB",
            self.app_version, self.mac.macos_version, self.mac.chip_name, self.mac.memory_gb
        )
    }

    /// The envelope sent to the receiver. Every count shares one fingerprint
    /// so the receiver lists them as one entry whose events are the counts,
    /// broken down by the four values as tags.
    pub(crate) fn envelope_body(&self, event_id: &str, sent_at_unix_seconds: u64) -> String {
        sentry_envelope_for_event(
            json!({
                "level": "info",
                "release": format!("astronomical@{}", self.app_version),
                "message": {"formatted": "daily count"},
                "fingerprint": ["daily_count"],
                "tags": {
                    "kind": "daily_count",
                    "chip": self.mac.chip_name,
                    "memory_gb": self.mac.memory_gb.to_string(),
                    "macos": self.mac.macos_version,
                },
            }),
            event_id,
            sent_at_unix_seconds,
        )
    }
}

#[cfg(test)]
#[path = "daily_count_line_tests.rs"]
mod daily_count_line_tests;
