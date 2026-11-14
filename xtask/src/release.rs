//! `cargo xtask release <step>` — the release-only steps (disk image,
//! notarization, distribution checks, installation, publication) as typed
//! commands, ported one by one from `scripts/release/`. Nothing here runs
//! during ordinary development, commits, or pull requests; CI exercises
//! only the pure planning and checking functions through unit tests.

mod app_store_package;
mod app_store_upload;
mod command;
mod disk_image_creation;
mod disk_image_validation;
mod distribution_signature;
mod github_publication;
mod notarization;
mod publication_entry;
mod release_identity;
mod release_manifest;
mod release_preparation;
mod scratch_space;
mod sparkle_acceptance;
mod stable_installation;
mod system_step;

#[cfg(test)]
#[path = "release_installation_tests.rs"]
mod release_installation_tests;
#[cfg(test)]
#[path = "release_publication_tests.rs"]
mod release_publication_tests;
#[cfg(test)]
#[path = "release_tests.rs"]
mod release_tests;

pub use command::run;
