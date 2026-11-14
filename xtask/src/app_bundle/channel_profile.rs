//! What each distribution channel *is*, as data.
//!
//! The shell builder spread these facts over a `case` block and about ten
//! `if channel != app-store` checks, so reading one channel meant reading the
//! whole script. Here every channel is one value, and the assembly code asks
//! the profile questions instead of comparing channel names.

/// The main app's executable in `Contents/MacOS`. Not "Astronomical": the
/// terminal command there is `astronomical`, and the default Mac file system
/// treats names that differ only in case as the same file.
pub(crate) const APPLICATION_EXECUTABLE_NAME: &str = "astronomical-app";
/// The menu-bar agent's executable, and its SwiftPM product. One agent serves
/// every channel: it links neither Sparkle nor the chat window.
pub(crate) const AGENT_EXECUTABLE_NAME: &str = "astronomical-menu";
/// The engine the menu-bar agent starts.
pub(crate) const DAEMON_EXECUTABLE_NAME: &str = "astronomicald";
/// The process the engine runs models in.
pub(crate) const WORKER_EXECUTABLE_NAME: &str = "astronomical-inference-worker";
/// The terminal command a direct build offers to link into the user's path.
pub(crate) const TERMINAL_COMMAND_EXECUTABLE_NAME: &str = "astronomical";
/// Where macOS looks for a helper the main app registers to start at login.
/// Mirrors `loginItemsPathComponents` in the Swift `ApplicationBundleLayout`,
/// which is how each process finds the other at run time.
const LOGIN_ITEMS_DIRECTORY: &str = "Contents/Library/LoginItems";

/// A distribution channel of the macOS app.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Channel {
    Development,
    Stable,
    AppStore,
}

impl Channel {
    pub(crate) fn from_channel_name(channel_name: &str) -> Result<Self, String> {
        match channel_name {
            "development" => Ok(Self::Development),
            "stable" => Ok(Self::Stable),
            "app-store" => Ok(Self::AppStore),
            _ => Err(format!(
                "channel must be development, stable, or app-store (got {channel_name:?})"
            )),
        }
    }

    pub(crate) fn as_channel_name(self) -> &'static str {
        match self {
            Self::Development => "development",
            Self::Stable => "stable",
            Self::AppStore => "app-store",
        }
    }
}

/// How the bundle is signed. Development and Stable without an identity use
/// ad-hoc signatures; the App Store channel must carry sandbox entitlements
/// on every executable, because the sandbox is established from signature
/// entitlements rather than from bundle layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SandboxPolicy {
    Unsandboxed,
    AppStoreSandbox,
}

/// Every channel-specific fact the bundle assembly needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChannelProfile {
    pub(crate) channel: Channel,
    /// Directory under `target/` that holds the bundle; `.noindex` keeps
    /// generated apps out of Spotlight so only installed copies appear there.
    pub(crate) output_directory_name: &'static str,
    pub(crate) bundle_name: &'static str,
    pub(crate) bundle_identifier: &'static str,
    pub(crate) daemon_port: u16,
    /// `None` for the App Store, whose state derives from Application Support.
    pub(crate) state_directory_name: Option<&'static str>,
    pub(crate) icon_channel: &'static str,
    /// The scheme the Observatory opens chat through. Development's is its own,
    /// because macOS hands a scheme shared by installed apps to whichever it
    /// prefers, which let Development's Observatory open Stable's chat window.
    pub(crate) chat_url_scheme: &'static str,
    /// The main app's SwiftPM product: the direct channels link Sparkle, the
    /// App Store product does not.
    pub(crate) application_swift_product: &'static str,
    pub(crate) daemon_cargo_features: Option<&'static str>,
    /// Sparkle is structurally absent from the App Store build.
    pub(crate) ships_sparkle: bool,
    pub(crate) sandbox_policy: SandboxPolicy,
    /// The name of the app group the bundle's two applications share, without
    /// its Team ID prefix. Only a sandboxed channel has one: unsandboxed
    /// applications already see the same home directory.
    pub(crate) application_group_name: Option<&'static str>,
    /// Release channels refuse a dirty tree so a published build is always
    /// reproducible from its commit.
    pub(crate) requires_clean_worktree: bool,
    /// Whether the bundle carries the `astronomical` terminal command. The
    /// store build leaves it out: it runs from the user's terminal, so it
    /// cannot inherit the app's sandbox, and App Store Connect refuses a
    /// bundle holding an executable that is not sandboxed (issue #2183).
    pub(crate) bundles_terminal_command: bool,
}

impl ChannelProfile {
    pub(crate) fn for_channel(channel: Channel) -> Self {
        match channel {
            Channel::Development => development_profile(),
            Channel::Stable => stable_profile(),
            Channel::AppStore => app_store_profile(),
        }
    }

    /// The Rust executables this channel's bundle carries in `Contents/MacOS`.
    pub(crate) fn bundled_rust_executable_names(&self) -> Vec<&'static str> {
        let mut bundled_executable_names = vec![DAEMON_EXECUTABLE_NAME, WORKER_EXECUTABLE_NAME];
        if self.bundles_terminal_command {
            bundled_executable_names.push(TERMINAL_COMMAND_EXECUTABLE_NAME);
        }
        bundled_executable_names
    }

    pub(crate) fn bundle_directory_name(&self) -> String {
        format!("{}.app", self.bundle_name)
    }

    /// The agent's name in Login Items settings and in Activity Monitor.
    pub(crate) fn agent_bundle_name(&self) -> String {
        format!("{} Agent", self.bundle_name)
    }

    /// A helper's identifier must extend its main app's for the App Store,
    /// and the main app registers the agent at login under this exact value.
    /// Mirrors `agentBundleIdentifier` in the Swift `ApplicationChannel`.
    pub(crate) fn agent_bundle_identifier(&self) -> String {
        format!("{}.agent", self.bundle_identifier)
    }

    /// The agent bundle's path inside the main bundle.
    pub(crate) fn agent_bundle_relative_path(&self) -> String {
        format!("{LOGIN_ITEMS_DIRECTORY}/{}.app", self.agent_bundle_name())
    }

    /// The agent executable's path inside the main bundle.
    pub(crate) fn agent_executable_relative_path(&self) -> String {
        format!(
            "{}/Contents/MacOS/{AGENT_EXECUTABLE_NAME}",
            self.agent_bundle_relative_path()
        )
    }
}

fn development_profile() -> ChannelProfile {
    ChannelProfile {
        channel: Channel::Development,
        output_directory_name: "astronomical-macos-development.noindex",
        bundle_name: "Astronomical Development",
        bundle_identifier: "dev.astronomical.app.development",
        daemon_port: 6733,
        state_directory_name: Some(".astronomical-dev"),
        icon_channel: "development",
        chat_url_scheme: "astronomical-dev",
        application_swift_product: "astronomical-app",
        daemon_cargo_features: None,
        ships_sparkle: true,
        sandbox_policy: SandboxPolicy::Unsandboxed,
        application_group_name: None,
        requires_clean_worktree: false,
        bundles_terminal_command: true,
    }
}

fn stable_profile() -> ChannelProfile {
    ChannelProfile {
        channel: Channel::Stable,
        output_directory_name: "astronomical-macos-stable.noindex",
        bundle_name: "Astronomical",
        bundle_identifier: "dev.astronomical.app",
        daemon_port: 6732,
        state_directory_name: Some(".astronomical"),
        icon_channel: "stable",
        chat_url_scheme: "astronomical",
        application_swift_product: "astronomical-app",
        daemon_cargo_features: None,
        ships_sparkle: true,
        sandbox_policy: SandboxPolicy::Unsandboxed,
        application_group_name: None,
        requires_clean_worktree: true,
        bundles_terminal_command: true,
    }
}

/// The Mac App Store variant of the Stable product: same name, port, and
/// icon, a store bundle identifier, no Sparkle, and a sandboxed state root.
fn app_store_profile() -> ChannelProfile {
    ChannelProfile {
        channel: Channel::AppStore,
        output_directory_name: "astronomical-macos-app-store.noindex",
        bundle_name: "Astronomical",
        bundle_identifier: "app.astronomical.appstore",
        daemon_port: 6732,
        state_directory_name: None,
        icon_channel: "stable",
        chat_url_scheme: "astronomical",
        application_swift_product: "astronomical-app-app-store",
        daemon_cargo_features: Some("astronomical-daemon/app-store-state-root"),
        ships_sparkle: false,
        sandbox_policy: SandboxPolicy::AppStoreSandbox,
        application_group_name: Some("astronomical"),
        requires_clean_worktree: true,
        bundles_terminal_command: false,
    }
}
