use std::{env, net::SocketAddr, path::PathBuf, str::FromStr};

use crate::AstronomicalConfigError;

const STABLE_STATE_DIRECTORY_NAME: &str = ".astronomical";
const DEVELOPMENT_STATE_DIRECTORY_NAME: &str = ".astronomical-dev";
const STABLE_BIND_ADDRESS: &str = "127.0.0.1:6732";
const DEVELOPMENT_BIND_ADDRESS: &str = "127.0.0.1:6733";
// App Store channel state roots. Sandboxed apps may write only inside their
// container, and the platform-standard Application Support directory is mapped
// into that container automatically, so the store build derives all state from
// it instead of a home-directory dot-folder (App Review guideline 2.4.5(ii)).
const APPLICATION_SUPPORT_STABLE_DIRECTORY_NAME: &str = "Astronomical";
const APPLICATION_SUPPORT_DEVELOPMENT_DIRECTORY_NAME: &str = "Astronomical Development";

/// User-visible runtime identity that keeps Stable and Development state apart.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AstronomicalRuntimeInstance {
    Stable,
    Development,
}

impl AstronomicalRuntimeInstance {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Development => "development",
        }
    }

    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Stable => "Stable",
            Self::Development => "Development",
        }
    }
}

impl FromStr for AstronomicalRuntimeInstance {
    type Err = AstronomicalConfigError;

    fn from_str(raw_instance: &str) -> Result<Self, Self::Err> {
        match raw_instance {
            "stable" => Ok(Self::Stable),
            "development" => Ok(Self::Development),
            _ => Err(AstronomicalConfigError::InvalidRuntimeInstance {
                raw_instance: raw_instance.to_owned(),
            }),
        }
    }
}

/// Complete writable path and endpoint boundary for one Astronomical instance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AstronomicalInstancePaths {
    runtime_instance: Option<AstronomicalRuntimeInstance>,
    state_directory: PathBuf,
    default_bind_address: SocketAddr,
    is_standard_state_directory: bool,
}

impl AstronomicalInstancePaths {
    pub fn for_current_user(
        runtime_instance: AstronomicalRuntimeInstance,
    ) -> Result<Self, AstronomicalConfigError> {
        let home_directory = env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or(AstronomicalConfigError::HomeDirectoryRequired)?;
        Self::for_user_home_directory(home_directory, runtime_instance)
    }

    /// Resolves an active user's existing home before deriving standard state.
    /// Acceptance fixtures use `for_home_directory` when no real home exists.
    pub fn for_user_home_directory(
        home_directory: impl Into<PathBuf>,
        runtime_instance: AstronomicalRuntimeInstance,
    ) -> Result<Self, AstronomicalConfigError> {
        let configured_home_directory = home_directory.into();
        if !configured_home_directory.is_absolute() {
            return Err(AstronomicalConfigError::PathMustBeAbsolute {
                field_name: "HOME".to_owned(),
                configured_path: configured_home_directory,
            });
        }
        let canonical_home_directory =
            configured_home_directory.canonicalize().map_err(|source| {
                AstronomicalConfigError::ResolveHomeDirectory {
                    home_directory: configured_home_directory,
                    source,
                }
            })?;
        if canonical_home_directory.parent().is_none() {
            return Err(AstronomicalConfigError::HomeDirectoryMustNotBeRoot);
        }
        Ok(Self::for_home_directory(
            canonical_home_directory,
            runtime_instance,
        ))
    }

    /// Instance paths for the default user location of `runtime_instance`.
    ///
    /// This is the single channel switch point. Direct-channel builds resolve
    /// beneath the home-directory dot-folder; builds compiled with the
    /// `app-store-state-root` feature resolve beneath the platform-standard
    /// macOS Application Support directory, which the sandbox maps into the
    /// app container. Every default-location caller delegates here so the two
    /// channels cannot disagree about state placement.
    pub fn default_location_instance_paths(
        runtime_instance: AstronomicalRuntimeInstance,
    ) -> Result<Self, AstronomicalConfigError> {
        #[cfg(feature = "app-store-state-root")]
        {
            let application_support_directory = Self::macos_application_support_directory()?;
            Ok(Self::for_application_support_directory(
                application_support_directory,
                runtime_instance,
            ))
        }
        #[cfg(not(feature = "app-store-state-root"))]
        {
            Self::for_current_user(runtime_instance)
        }
    }

    #[must_use]
    pub fn for_home_directory(
        home_directory: impl Into<PathBuf>,
        runtime_instance: AstronomicalRuntimeInstance,
    ) -> Self {
        let state_directory_name = match runtime_instance {
            AstronomicalRuntimeInstance::Stable => STABLE_STATE_DIRECTORY_NAME,
            AstronomicalRuntimeInstance::Development => DEVELOPMENT_STATE_DIRECTORY_NAME,
        };
        Self::for_state_directory_with_standard_endpoint(
            home_directory.into().join(state_directory_name),
            runtime_instance,
        )
    }

    /// Resolves the platform-standard macOS Application Support directory for
    /// the active user. Inside the App Sandbox this path is mapped into the
    /// app container on every file operation, so the same resolution serves
    /// both the direct and the App Store channel.
    pub fn macos_application_support_directory() -> Result<PathBuf, AstronomicalConfigError> {
        let home_directory = env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or(AstronomicalConfigError::HomeDirectoryRequired)?;
        if !home_directory.is_absolute() {
            return Err(AstronomicalConfigError::PathMustBeAbsolute {
                field_name: "HOME".to_owned(),
                configured_path: home_directory,
            });
        }
        let canonical_home_directory = home_directory.canonicalize().map_err(|source| {
            AstronomicalConfigError::ResolveHomeDirectory {
                home_directory,
                source,
            }
        })?;
        Ok(canonical_home_directory.join("Library/Application Support"))
    }

    /// Resolves instance state beneath the platform-standard macOS Application
    /// Support directory. The App Store channel uses this instead of the
    /// home-directory dot-folder because sandboxed store builds may write only
    /// inside their container, and Application Support is the container-mapped
    /// location Apple's file-system requirements name for persistent state.
    /// Standard-instance semantics (loopback endpoint guards) carry over
    /// unchanged so the store build keeps the same endpoint discipline as the
    /// direct channel.
    #[must_use]
    pub fn for_application_support_directory(
        application_support_directory: impl Into<PathBuf>,
        runtime_instance: AstronomicalRuntimeInstance,
    ) -> Self {
        let state_directory_name = match runtime_instance {
            AstronomicalRuntimeInstance::Stable => APPLICATION_SUPPORT_STABLE_DIRECTORY_NAME,
            AstronomicalRuntimeInstance::Development => {
                APPLICATION_SUPPORT_DEVELOPMENT_DIRECTORY_NAME
            }
        };
        Self::for_state_directory_with_standard_endpoint(
            application_support_directory
                .into()
                .join(state_directory_name),
            runtime_instance,
        )
    }

    #[must_use]
    fn for_state_directory_with_standard_endpoint(
        state_directory: PathBuf,
        runtime_instance: AstronomicalRuntimeInstance,
    ) -> Self {
        let default_bind_address = match runtime_instance {
            AstronomicalRuntimeInstance::Stable => STABLE_BIND_ADDRESS,
            AstronomicalRuntimeInstance::Development => DEVELOPMENT_BIND_ADDRESS,
        }
        .parse()
        .expect("built-in Astronomical loopback addresses must remain valid");
        Self {
            runtime_instance: Some(runtime_instance),
            state_directory,
            default_bind_address,
            is_standard_state_directory: true,
        }
    }

    #[must_use]
    pub fn for_state_directory(
        state_directory: PathBuf,
        runtime_instance: AstronomicalRuntimeInstance,
    ) -> Self {
        // Custom state must coexist with installed channels and parallel test instances without
        // restoring a user-editable endpoint to the strict public configuration document.
        let default_bind_address = SocketAddr::from(([127, 0, 0, 1], 0));
        Self {
            runtime_instance: Some(runtime_instance),
            state_directory,
            default_bind_address,
            is_standard_state_directory: false,
        }
    }

    #[must_use]
    pub const fn for_explicit_state_directory(
        state_directory: PathBuf,
        default_bind_address: SocketAddr,
    ) -> Self {
        Self {
            runtime_instance: None,
            state_directory,
            default_bind_address,
            is_standard_state_directory: false,
        }
    }

    #[must_use]
    pub const fn runtime_instance(&self) -> Option<AstronomicalRuntimeInstance> {
        self.runtime_instance
    }

    #[must_use]
    pub fn state_directory(&self) -> &std::path::Path {
        &self.state_directory
    }

    #[must_use]
    pub const fn default_bind_address(&self) -> SocketAddr {
        self.default_bind_address
    }

    #[must_use]
    pub const fn is_standard_state_directory(&self) -> bool {
        self.is_standard_state_directory
    }

    /// Prevents a standard Stable or Development instance from adopting the
    /// other channel's endpoint while leaving explicit test state configurable.
    pub fn validate_configured_bind_address(
        &self,
        configured_bind_address: SocketAddr,
    ) -> Result<SocketAddr, AstronomicalConfigError> {
        if self.is_standard_state_directory && configured_bind_address != self.default_bind_address
        {
            return Err(
                AstronomicalConfigError::StandardInstanceBindAddressMismatch {
                    configured_bind_address,
                    expected_bind_address: self.default_bind_address,
                },
            );
        }
        Ok(configured_bind_address)
    }

    #[must_use]
    pub fn config_file_path(&self) -> PathBuf {
        self.state_directory.join("config.json")
    }

    #[must_use]
    pub fn prompt_cache_directory(&self) -> PathBuf {
        self.state_directory.join("cache")
    }

    #[must_use]
    pub fn models_directory(&self) -> PathBuf {
        self.state_directory.join("models")
    }

    #[must_use]
    pub fn logging_directory(&self) -> PathBuf {
        self.state_directory.join("logs")
    }

    #[must_use]
    pub fn daemon_ownership_file_path(&self) -> PathBuf {
        self.state_directory.join("menu-owned-daemon.json")
    }

    #[must_use]
    pub fn instance_lock_file_path(&self) -> PathBuf {
        self.state_directory.join("instance.lock")
    }

    /// Optional user-authored Markdown seeded into Qwen3.5 reasoning.
    ///
    /// The file is never created automatically. Missing is a no-op at request time.
    #[must_use]
    pub fn qwen_thinking_channel_seed_file_path(&self) -> PathBuf {
        self.state_directory.join("thinking.md")
    }
}
