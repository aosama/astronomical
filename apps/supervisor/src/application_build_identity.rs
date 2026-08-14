use serde::Serialize;

/// Build provenance shown through every Astronomical control surface.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ApplicationBuildIdentity {
    pub version: &'static str,
    pub build_number: u64,
    pub commit: &'static str,
    pub is_dirty: bool,
}

impl ApplicationBuildIdentity {
    #[must_use]
    pub fn current() -> Self {
        let build_number = option_env!("ASTRONOMICAL_BUILD_NUMBER")
            .and_then(|raw_build_number| raw_build_number.parse::<u64>().ok())
            .unwrap_or(0);
        Self {
            version: env!("CARGO_PKG_VERSION"),
            build_number,
            commit: option_env!("ASTRONOMICAL_BUILD_COMMIT").unwrap_or("unknown"),
            is_dirty: option_env!("ASTRONOMICAL_BUILD_DIRTY") == Some("true"),
        }
    }

    #[must_use]
    pub fn command_line_version(&self) -> String {
        let dirty_suffix = if self.is_dirty { "-dirty" } else { "" };
        format!(
            "astronomicald {} (build {}, {}{})",
            self.version, self.build_number, self.commit, dirty_suffix
        )
    }
}
