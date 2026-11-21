//! The skill switches as the daemon uses them: resolved from config.json and
//! the instance's paths once per reload, so serving a request never touches
//! the disk or the config file to decide whether to declare the skill.

use std::path::PathBuf;

/// How the Astronomical skill reaches models and stays current, for one
/// resolved configuration.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SkillSettings {
    /// Whether every chat request is told the skill exists
    /// (`skill.declare_to_models`). Applied live.
    pub declared_to_models: bool,
    /// Whether the daemon looks for a newer skill on the project's site
    /// (`skill.update_from_web`). Applied live.
    pub updates_from_web: bool,
    /// The skill's index file, the path a declaration tells models to read.
    pub index_path: PathBuf,
}

impl SkillSettings {
    /// The default, which leaves every request untouched, is what an
    /// executor without a resolved configuration gets, such as a test harness.
    #[must_use]
    pub fn resolve(
        user_config: &config::AstronomicalConfig,
        skill_directory: &std::path::Path,
    ) -> Self {
        Self {
            declared_to_models: user_config.skill_declared_to_models(),
            updates_from_web: user_config.skill_updates_from_web(),
            index_path: crate::astronomical_skill::skill_index_path(skill_directory),
        }
    }
}

#[cfg(test)]
#[path = "skill_settings_tests.rs"]
mod skill_settings_tests;
