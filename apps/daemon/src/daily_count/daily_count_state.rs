//! What the daily count remembers on this Mac: the first day the engine ran,
//! so the question waits until the second day, and the last day a count was
//! sent, so a restart never sends twice in a day.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// The state file's name inside the instance's state folder.
pub(crate) const DAILY_COUNT_STATE_FILE_NAME: &str = "daily-count-state.json";

const SECONDS_PER_DAY: u64 = 86_400;

/// Days since 1970-01-01 in UTC. One shared clock for every Mac keeps "a
/// day" the same length everywhere and needs no time zone.
pub(crate) fn unix_day_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since_epoch| since_epoch.as_secs() / SECONDS_PER_DAY)
        .unwrap_or_default()
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct DailyCountState {
    pub(crate) first_used_unix_day: u64,
    pub(crate) last_sent_unix_day: Option<u64>,
}

impl DailyCountState {
    fn file_path(state_directory: &Path) -> PathBuf {
        state_directory.join(DAILY_COUNT_STATE_FILE_NAME)
    }

    /// Reads the state, or starts it at `today` when there is none or it
    /// cannot be read. Starting again only delays the question by a day.
    pub(crate) fn load_or_start(state_directory: &Path, today_unix_day: u64) -> Self {
        let stored_state = fs::read(Self::file_path(state_directory))
            .ok()
            .and_then(|state_bytes| serde_json::from_slice::<Self>(&state_bytes).ok());
        if let Some(stored_state) = stored_state {
            return stored_state;
        }
        let started_state = Self {
            first_used_unix_day: today_unix_day,
            last_sent_unix_day: None,
        };
        // A state that cannot be saved is started again next time, which
        // only delays the question; it never causes a count to be sent.
        let _ = started_state.save(state_directory);
        started_state
    }

    pub(crate) fn save(&self, state_directory: &Path) -> std::io::Result<()> {
        fs::create_dir_all(state_directory)?;
        let state_bytes = serde_json::to_vec(self).map_err(std::io::Error::other)?;
        fs::write(Self::file_path(state_directory), state_bytes)
    }

    /// Whether to ask "May this Mac be counted?": only while unanswered, and
    /// only from the day after the first, when the app has shown it works.
    pub(crate) fn should_ask(&self, daily_count_answer: Option<bool>, today_unix_day: u64) -> bool {
        daily_count_answer.is_none() && today_unix_day > self.first_used_unix_day
    }

    pub(crate) fn is_sent_today(&self, today_unix_day: u64) -> bool {
        self.last_sent_unix_day == Some(today_unix_day)
    }
}

#[cfg(test)]
#[path = "daily_count_state_tests.rs"]
mod daily_count_state_tests;
