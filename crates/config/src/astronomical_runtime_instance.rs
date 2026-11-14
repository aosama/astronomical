use std::{env, net::SocketAddr, path::PathBuf, str::FromStr};

use crate::AstronomicalConfigError;

const STABLE_STATE_DIRECTORY_NAME: &str = ".astronomical";
const DEVELOPMENT_STATE_DIRECTORY_NAME: &str = ".astronomical-dev";
const STABLE_LOOPBACK_PORT: u16 = 6732;
const DEVELOPMENT_LOOPBACK_PORT: u16 = 6733;
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

    /// Loopback listener owned by this runtime instance.
    #[must_use]
    pub fn loopback_socket_addr(self) -> SocketAddr {
        match self {
            Self::Stable => SocketAddr::from(([127, 0, 0, 1], STABLE_LOOPBACK_PORT)),
            Self::Development => SocketAddr::from(([127, 0, 0, 1], DEVELOPMENT_LOOPBACK_PORT)),
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
    state_directory_label: StateDirectoryLabel,
}

/// How status names the state directory to the applications that check they
/// reached their own engine.
#[derive(Clone, Debug, Eq, PartialEq)]
enum StateDirectoryLabel {
    /// A standard location, written from `~` so status never carries the
    /// user's absolute home directory.
    HomeRelative(String),
    /// A caller-chosen directory, which no installed application expects.
    Custom,
}

/// The label of a state directory no installed application expects.
pub const CUSTOM_STATE_DIRECTORY_LABEL: &str = "custom";
/// Where the sandbox maps Application Support, written from `~`.
const APPLICATION_SUPPORT_LABEL_ROOT: &str = "~/Library/Application Support";
/// Where macOS keeps every app group container, written from `~`.
const APPLICATION_GROUP_CONTAINERS_LABEL_ROOT: &str = "~/Library/Group Containers";

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
            format!("~/{state_directory_name}"),
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
        Self::for_container_state_root(
            application_support_directory.into(),
            APPLICATION_SUPPORT_LABEL_ROOT,
            runtime_instance,
        )
    }

    /// Resolves instance state beneath an app group container.
    ///
    /// The App Store build runs as two sandboxed applications, the chat app
    /// and the menu-bar agent that owns the engine, and each has a container
    /// of its own. A group container is the one location both may open, so
    /// the engine's state lives there; in a single app's container the other
    /// application could never reach it. The caller supplies the container
    /// because only the platform can resolve it (it is named after the
    /// signing team), and the directory names match the Application Support
    /// layout so a state root can be moved between the two unchanged.
    #[must_use]
    pub fn for_application_group_container(
        application_group_container: impl Into<PathBuf>,
        runtime_instance: AstronomicalRuntimeInstance,
    ) -> Self {
        let application_group_container = application_group_container.into();
        // The container's folder is named after the group, which carries the
        // signing team, so the label can only be read off the path it was given.
        let label_root = match application_group_container.file_name() {
            Some(group_folder_name) => format!(
                "{APPLICATION_GROUP_CONTAINERS_LABEL_ROOT}/{}",
                group_folder_name.to_string_lossy()
            ),
            None => APPLICATION_GROUP_CONTAINERS_LABEL_ROOT.to_owned(),
        };
        Self::for_container_state_root(application_group_container, &label_root, runtime_instance)
    }

    /// The Application Support layout, shared by a container's own
    /// Application Support and an app group container: the same directory
    /// names beneath either root, labeled with the root's `~` form.
    fn for_container_state_root(
        state_root_directory: PathBuf,
        label_root: &str,
        runtime_instance: AstronomicalRuntimeInstance,
    ) -> Self {
        let state_directory_name = match runtime_instance {
            AstronomicalRuntimeInstance::Stable => APPLICATION_SUPPORT_STABLE_DIRECTORY_NAME,
            AstronomicalRuntimeInstance::Development => {
                APPLICATION_SUPPORT_DEVELOPMENT_DIRECTORY_NAME
            }
        };
        Self::for_state_directory_with_standard_endpoint(
            state_root_directory.join(state_directory_name),
            runtime_instance,
            format!("{label_root}/{state_directory_name}"),
        )
    }

    #[must_use]
    fn for_state_directory_with_standard_endpoint(
        state_directory: PathBuf,
        runtime_instance: AstronomicalRuntimeInstance,
        home_relative_label: String,
    ) -> Self {
        let default_bind_address = runtime_instance.loopback_socket_addr();
        Self {
            runtime_instance: Some(runtime_instance),
            state_directory,
            default_bind_address,
            is_standard_state_directory: true,
            state_directory_label: StateDirectoryLabel::HomeRelative(home_relative_label),
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
            state_directory_label: StateDirectoryLabel::Custom,
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
            state_directory_label: StateDirectoryLabel::Custom,
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

    /// What status reports as this instance's state directory. The chat app
    /// and the menu-bar agent compare it with the location they expect, so it
    /// names where the state really is: a store build's engine that reported
    /// the direct build's dot-folder was refused by its own agent and
    /// restarted every few seconds (issue #2188).
    #[must_use]
    pub fn state_directory_label(&self) -> &str {
        match &self.state_directory_label {
            StateDirectoryLabel::HomeRelative(home_relative_label) => home_relative_label,
            StateDirectoryLabel::Custom => CUSTOM_STATE_DIRECTORY_LABEL,
        }
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

    /// Where this instance keeps compiled Neural Engine programs unless the config says
    /// otherwise. Each instance keeps its own, so one instance's builds can never affect
    /// another's.
    #[must_use]
    pub fn default_neural_engine_program_directory(&self) -> PathBuf {
        self.state_directory
            .join(crate::NEURAL_ENGINE_PROGRAM_DIRECTORY_NAME)
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

    /// Private local IPC socket the daemon serves ephemeral CLI verbs on.
    #[must_use]
    pub fn ipc_socket_file_path(&self) -> PathBuf {
        self.state_directory.join("ipc.sock")
    }
}
