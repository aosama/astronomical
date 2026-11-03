//! What each distribution channel *is*, as data.
//!
//! The shell builder spread these facts over a `case` block and about ten
//! `if channel != app-store` checks, so reading one channel meant reading the
//! whole script. Here every channel is one value, and the assembly code asks
//! the profile questions instead of comparing channel names.

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
    pub(crate) menu_swift_product: &'static str,
    pub(crate) daemon_cargo_features: Option<&'static str>,
    /// Sparkle is structurally absent from the App Store build.
    pub(crate) ships_sparkle: bool,
    pub(crate) sandbox_policy: SandboxPolicy,
    /// Release channels refuse a dirty tree so a published build is always
    /// reproducible from its commit.
    pub(crate) requires_clean_worktree: bool,
}

impl ChannelProfile {
    pub(crate) fn for_channel(channel: Channel) -> Self {
        match channel {
            Channel::Development => development_profile(),
            Channel::Stable => stable_profile(),
            Channel::AppStore => app_store_profile(),
        }
    }

    pub(crate) fn bundle_directory_name(&self) -> String {
        format!("{}.app", self.bundle_name)
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
        menu_swift_product: "astronomical-menu",
        daemon_cargo_features: None,
        ships_sparkle: true,
        sandbox_policy: SandboxPolicy::Unsandboxed,
        requires_clean_worktree: false,
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
        menu_swift_product: "astronomical-menu",
        daemon_cargo_features: None,
        ships_sparkle: true,
        sandbox_policy: SandboxPolicy::Unsandboxed,
        requires_clean_worktree: true,
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
        menu_swift_product: "astronomical-menu-app-store",
        daemon_cargo_features: Some("astronomical-daemon/app-store-state-root"),
        ships_sparkle: false,
        sandbox_policy: SandboxPolicy::AppStoreSandbox,
        requires_clean_worktree: true,
    }
}
