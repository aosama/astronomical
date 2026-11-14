//! The two `Info.plist` files of the bundle, rendered from a channel profile
//! and a build identity: the main app's, and the menu-bar agent's.
//!
//! The main app is a regular application, so its plist carries no
//! `LSUIElement`. The agent's sets it, which makes the agent an accessory for
//! its whole life: neither process ever changes role while running, which is
//! what keeps the main app's menu bar reachable. Both carry the same
//! `Astronomical*` keys, because each reads its instance identity from its own
//! bundle. Rendering is pure, so every channel's plists are unit-tested
//! without building anything.

use super::build_identity::BuildIdentity;
use super::channel_profile::{AGENT_EXECUTABLE_NAME, APPLICATION_EXECUTABLE_NAME, ChannelProfile};

/// The primary App Store category of the product. It was first Developer
/// Tools (issue #492) and became Productivity when the store listing was
/// filled in, with Developer Tools as the secondary category there; the
/// plist names only the primary one, and it must match the listing.
pub(crate) const APPLICATION_CATEGORY: &str = "public.app-category.productivity";

/// Sparkle's EdDSA public key: updates must be signed by its private half.
pub(crate) const SPARKLE_PUBLIC_ED25519_KEY: &str = "g0URwy+j86uDYcmOu0k/IUVWwCOSrGOPSoFnVoYQ9AQ=";

pub(crate) fn render_info_plist(
    channel_profile: &ChannelProfile,
    build_identity: &BuildIdentity,
    update_feed_url: &str,
) -> String {
    let mut plist_text = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
         <plist version=\"1.0\">\n<dict>\n",
    );
    push_bundle_keys(
        &mut plist_text,
        &BundleNaming {
            bundle_name: channel_profile.bundle_name,
            bundle_identifier: channel_profile.bundle_identifier,
            executable_name: APPLICATION_EXECUTABLE_NAME,
        },
        build_identity,
    );
    if channel_profile.ships_sparkle {
        push_sparkle_keys(&mut plist_text, update_feed_url);
    }
    push_astronomical_keys(&mut plist_text, channel_profile, build_identity);
    // The Observatory console hands chat over through this channel's own scheme.
    plist_text.push_str(&format!(
        "  <key>CFBundleURLTypes</key>\n  <array>\n    <dict>\n      \
         <key>CFBundleURLName</key><string>Astronomical Thin Talk</string>\n      \
         <key>CFBundleURLSchemes</key>\n      <array>\n        \
         <string>{}</string>\n      </array>\n    </dict>\n  </array>\n",
        channel_profile.chat_url_scheme,
    ));
    push_string(&mut plist_text, "CFBundlePackageType", "APPL");
    // App Store Connect refuses a Mac app without a category, and the same
    // key files a direct install under Productivity in Launchpad.
    push_string(
        &mut plist_text,
        "LSApplicationCategoryType",
        APPLICATION_CATEGORY,
    );
    plist_text.push_str("</dict>\n</plist>\n");
    plist_text
}

/// The agent's plist: no URL scheme (links belong to the main app) and no
/// Sparkle keys (the main app owns updates).
pub(crate) fn render_agent_info_plist(
    channel_profile: &ChannelProfile,
    build_identity: &BuildIdentity,
) -> String {
    let mut plist_text = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
         <plist version=\"1.0\">\n<dict>\n",
    );
    push_bundle_keys(
        &mut plist_text,
        &BundleNaming {
            bundle_name: &channel_profile.agent_bundle_name(),
            bundle_identifier: &channel_profile.agent_bundle_identifier(),
            executable_name: AGENT_EXECUTABLE_NAME,
        },
        build_identity,
    );
    push_astronomical_keys(&mut plist_text, channel_profile, build_identity);
    push_string(&mut plist_text, "CFBundlePackageType", "APPL");
    push_boolean(&mut plist_text, "LSUIElement", true);
    plist_text.push_str("</dict>\n</plist>\n");
    plist_text
}

/// What tells the main app's bundle keys from the agent's.
struct BundleNaming<'naming> {
    bundle_name: &'naming str,
    bundle_identifier: &'naming str,
    executable_name: &'naming str,
}

fn push_bundle_keys(
    plist_text: &mut String,
    bundle_naming: &BundleNaming<'_>,
    build_identity: &BuildIdentity,
) {
    push_string(plist_text, "CFBundleName", bundle_naming.bundle_name);
    push_string(plist_text, "CFBundleDisplayName", bundle_naming.bundle_name);
    push_string(
        plist_text,
        "CFBundleExecutable",
        bundle_naming.executable_name,
    );
    push_string(
        plist_text,
        "CFBundleIdentifier",
        bundle_naming.bundle_identifier,
    );
    push_string(plist_text, "CFBundleVersion", &build_identity.build_number);
    push_string(
        plist_text,
        "CFBundleShortVersionString",
        &build_identity.application_version,
    );
    push_string(plist_text, "LSMinimumSystemVersion", "14.0");
    push_string(plist_text, "CFBundleIconFile", "Astronomical.icns");
}

/// Updates are verified before extraction and the feed itself must be
/// signed; a signed-feed failure never expires into trusting an unsigned one.
fn push_sparkle_keys(plist_text: &mut String, update_feed_url: &str) {
    push_string(plist_text, "SUFeedURL", update_feed_url);
    push_string(plist_text, "SUPublicEDKey", SPARKLE_PUBLIC_ED25519_KEY);
    push_boolean(plist_text, "SUVerifyUpdateBeforeExtraction", true);
    push_boolean(plist_text, "SURequireSignedFeed", true);
    push_integer(plist_text, "SUSignedFeedFailureExpirationInterval", 0);
    push_boolean(plist_text, "SUEnableAutomaticChecks", true);
    push_integer(plist_text, "SUScheduledCheckInterval", 86_400);
    push_boolean(plist_text, "SUAutomaticallyUpdate", false);
    push_boolean(plist_text, "SUEnableSystemProfiling", false);
}

fn push_astronomical_keys(
    plist_text: &mut String,
    channel_profile: &ChannelProfile,
    build_identity: &BuildIdentity,
) {
    push_string(
        plist_text,
        "AstronomicalChannel",
        channel_profile.channel.as_channel_name(),
    );
    push_string(
        plist_text,
        "AstronomicalBuildDate",
        &build_identity.build_date_utc,
    );
    push_integer(
        plist_text,
        "AstronomicalDaemonPort",
        u64::from(channel_profile.daemon_port),
    );
    if let Some(state_directory_name) = channel_profile.state_directory_name {
        push_string(
            plist_text,
            "AstronomicalStateDirectoryName",
            state_directory_name,
        );
    }
    push_string(
        plist_text,
        "AstronomicalBuildCommit",
        &build_identity.build_commit,
    );
    push_boolean(
        plist_text,
        "AstronomicalBuildDirty",
        build_identity.is_dirty,
    );
}

fn push_string(plist_text: &mut String, plist_key: &str, string_value: &str) {
    plist_text.push_str(&format!(
        "  <key>{plist_key}</key><string>{}</string>\n",
        escape_xml_text(string_value)
    ));
}

fn push_integer(plist_text: &mut String, plist_key: &str, integer_value: u64) {
    plist_text.push_str(&format!(
        "  <key>{plist_key}</key><integer>{integer_value}</integer>\n"
    ));
}

fn push_boolean(plist_text: &mut String, plist_key: &str, boolean_value: bool) {
    let boolean_element = if boolean_value { "<true/>" } else { "<false/>" };
    plist_text.push_str(&format!("  <key>{plist_key}</key>{boolean_element}\n"));
}

/// A feed URL or a future bundle name may carry `&` or `<`; the shell
/// builder's `printf` would have written them raw and broken the plist.
pub(crate) fn escape_xml_text(raw_text: &str) -> String {
    let mut escaped_text = String::with_capacity(raw_text.len());
    for raw_character in raw_text.chars() {
        match raw_character {
            '&' => escaped_text.push_str("&amp;"),
            '<' => escaped_text.push_str("&lt;"),
            '>' => escaped_text.push_str("&gt;"),
            '"' => escaped_text.push_str("&quot;"),
            _ => escaped_text.push(raw_character),
        }
    }
    escaped_text
}
