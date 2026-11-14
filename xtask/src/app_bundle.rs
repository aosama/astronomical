//! `cargo xtask app` — builds the macOS app bundle from typed channel
//! profiles. `app dev` is the Development edit loop: persistent incremental
//! outputs, cached icon and npm install, ad-hoc signing, structural checks,
//! then a swap of the running app. `app instance` controls that instance
//! headless. `app package` builds Stable and App Store bundles from a clean
//! commit in a disposable target directory, and `app smoke-test` launches a
//! built bundle in isolation.

mod application_group;
mod build_identity;
mod build_steps;
mod bundle_assembly;
mod bundle_check;
mod channel_profile;
mod code_signing;
mod command;
mod dev_pipeline;
mod icon_cache;
mod info_plist;
mod instance_control;
mod login_launch_check;
mod native_metallib;
mod package_pipeline;
mod release_build_steps;
mod running_instance;
mod smoke_test;
mod step_reporter;
mod tool_command;

#[cfg(test)]
#[path = "app_bundle_link_tests.rs"]
mod app_bundle_link_tests;
#[cfg(test)]
#[path = "app_bundle_tests.rs"]
mod app_bundle_tests;
#[cfg(test)]
#[path = "app_export_declaration_tests.rs"]
mod app_export_declaration_tests;
#[cfg(test)]
#[path = "app_login_launch_check_tests.rs"]
mod app_login_launch_check_tests;
#[cfg(test)]
#[path = "app_package_fixture_tests.rs"]
mod app_package_fixture_tests;
#[cfg(test)]
#[path = "app_package_group_tests.rs"]
mod app_package_group_tests;
#[cfg(test)]
#[path = "app_package_tests.rs"]
mod app_package_tests;
#[cfg(test)]
#[path = "app_package_two_process_tests.rs"]
mod app_package_two_process_tests;
#[cfg(test)]
#[path = "app_running_instance_tests.rs"]
mod app_running_instance_tests;

pub use command::run;
// Stable installation re-checks a bundle with the same rules that built it.
pub(crate) use bundle_check::{layout_problems, plist_identity_problems, version_text_reports};
// The distribution signature check names the same two Swift executables.
pub(crate) use channel_profile::{APPLICATION_EXECUTABLE_NAME, Channel, ChannelProfile};
// The publisher derives the release version and update feed exactly as the
// packager does, so the two cannot disagree about either.
pub(crate) use build_identity::{daemon_package_identity, github_pages_feed_url};
// The Sparkle acceptance journey builds fixture apps with the product's own
// update key and the same XML escaping.
pub(crate) use info_plist::{SPARKLE_PUBLIC_ED25519_KEY, escape_xml_text};
