//! The sharing choices as the daemon uses them: resolved from config.json once
//! per reload, so offering a problem report never reads the config file.

use config::ProblemReportChoice;

/// What the person chose to send to the developers, for one resolved
/// configuration. Both apply live; no worker reads them.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SharingSettings {
    /// How a problem report is handled (`sharing.problem_reports`).
    pub problem_report_choice: ProblemReportChoice,
    /// The answer to the daily count question, `None` while unasked
    /// (`sharing.daily_count`).
    pub daily_count_answer: Option<bool>,
}

impl SharingSettings {
    #[must_use]
    pub fn resolve(user_config: &config::AstronomicalConfig) -> Self {
        Self {
            problem_report_choice: user_config.problem_report_choice(),
            daily_count_answer: user_config.daily_count_answer(),
        }
    }

    /// The setting names that differ from `candidate`, in report order, for a
    /// reload's list of settings applied in place.
    #[must_use]
    pub(crate) fn changed_setting_names(&self, candidate: &Self) -> Vec<String> {
        let mut changed_setting_names = Vec::new();
        if self.problem_report_choice != candidate.problem_report_choice {
            changed_setting_names.push("sharing.problem_reports".to_owned());
        }
        if self.daily_count_answer != candidate.daily_count_answer {
            changed_setting_names.push("sharing.daily_count".to_owned());
        }
        changed_setting_names
    }
}

#[cfg(test)]
#[path = "sharing_settings_tests.rs"]
mod sharing_settings_tests;
