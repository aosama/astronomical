//! `cargo xtask release <step>` — the release-only steps (disk image,
//! notarization, distribution checks, installation, publication) as typed
//! commands, ported one by one from `scripts/release/`. Nothing here runs
//! during ordinary development, commits, or pull requests; CI exercises
//! only the pure planning and checking functions through unit tests.

mod app_store_package;
mod command;
mod disk_image_creation;
mod disk_image_validation;
mod distribution_signature;
mod notarization;
mod scratch_space;
mod stable_installation;
mod system_step;

#[cfg(test)]
#[path = "release_tests.rs"]
mod release_tests;

pub use command::run;
