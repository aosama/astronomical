//! The app group an App Store bundle's two applications share.
//!
//! The store build is two sandboxed applications, the chat app and the
//! menu-bar agent that owns the engine, and the sandbox gives each a container
//! of its own. A group container is the one location both may open, so the
//! engine's state lives there. A group is named after the signing team
//! (`TEAMID.name`), which is known only when a bundle is signed: that is why
//! the identifier is derived here, at packaging time, and written into both
//! plists and the sandbox entitlements rather than kept as a constant.

use super::channel_profile::ChannelProfile;

/// The plist key both applications read their group identifier from. Mirrors
/// the key `ApplicationIdentity.current` reads in the Swift shared module.
pub(crate) const APPLICATION_GROUP_PLIST_KEY: &str = "AstronomicalApplicationGroupIdentifier";
pub(crate) const APPLICATION_GROUPS_ENTITLEMENT_KEY: &str = "com.apple.security.application-groups";
/// Apple's Team IDs are ten uppercase letters and digits.
const TEAM_IDENTIFIER_LENGTH: usize = 10;
const PLIST_DICTIONARY_END: &str = "</dict>\n</plist>";

/// The Team ID in the final parentheses of a signing identity's name, such as
/// `Apple Distribution: Example Corp (TEAM123456)`.
pub(crate) fn team_identifier_of(signing_identity: &str) -> Option<&str> {
    let identity_without_closing = signing_identity.trim_end().strip_suffix(')')?;
    let (_, team_identifier) = identity_without_closing.rsplit_once('(')?;
    let is_team_identifier = team_identifier.len() == TEAM_IDENTIFIER_LENGTH
        && team_identifier
            .bytes()
            .all(|team_byte| team_byte.is_ascii_uppercase() || team_byte.is_ascii_digit());
    is_team_identifier.then_some(team_identifier)
}

/// The group identifier for a channel signed by `signing_identity`.
///
/// `Ok(None)` for a channel that is not sandboxed and so needs no group. A
/// sandboxed channel without a readable Team ID is refused: its two
/// applications would otherwise ship with separate state.
pub(crate) fn application_group_identifier(
    channel_profile: &ChannelProfile,
    signing_identity: Option<&str>,
) -> Result<Option<String>, String> {
    let Some(application_group_name) = channel_profile.application_group_name else {
        return Ok(None);
    };
    let team_identifier = match signing_identity.and_then(team_identifier_of) {
        Some(team_identifier) => team_identifier,
        None => {
            return Err(
                "the signing identity must end with its ten-character Team ID in parentheses, because the App Store build's app group is named after the team"
                    .to_owned(),
            );
        }
    };
    Ok(Some(format!("{team_identifier}.{application_group_name}")))
}

/// `plist_text` with the group identifier added as its last entry.
pub(crate) fn stamp_application_group(
    plist_text: &str,
    application_group_identifier: &str,
) -> Result<String, String> {
    insert_last_plist_entry(
        plist_text,
        &format!(
            "  <key>{APPLICATION_GROUP_PLIST_KEY}</key><string>{application_group_identifier}</string>\n"
        ),
    )
}

/// The sandbox entitlements of the two applications: the checked-in profile
/// plus membership of the group, which the checked-in file cannot name.
pub(crate) fn application_group_entitlements(
    sandbox_entitlements_text: &str,
    application_group_identifier: &str,
) -> Result<String, String> {
    insert_last_plist_entry(
        sandbox_entitlements_text,
        &format!(
            "    <key>{APPLICATION_GROUPS_ENTITLEMENT_KEY}</key>\n    <array>\n        <string>{application_group_identifier}</string>\n    </array>\n"
        ),
    )
}

/// Inserts `plist_entry` before the closing of the top-level dictionary. The
/// plists this is given are flat dictionaries written by this repository, so
/// the last `</dict>` is the top-level one.
fn insert_last_plist_entry(plist_text: &str, plist_entry: &str) -> Result<String, String> {
    let Some(dictionary_end_position) = plist_text.rfind(PLIST_DICTIONARY_END) else {
        return Err("the property list does not end with a top-level dictionary".to_owned());
    };
    let mut stamped_plist_text = String::with_capacity(plist_text.len() + plist_entry.len());
    stamped_plist_text.push_str(&plist_text[..dictionary_end_position]);
    stamped_plist_text.push_str(plist_entry);
    stamped_plist_text.push_str(&plist_text[dictionary_end_position..]);
    Ok(stamped_plist_text)
}
