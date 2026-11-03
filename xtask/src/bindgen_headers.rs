//! `cargo xtask provision-bindgen-headers` — provisions bindgen-ready MLX and
//! MLX-C headers from the verified native dependency archive cache, without
//! any CMake build step and without network access. Typed port of the
//! retired shell provisioning script and its sourced pipeline library; the
//! flags, step lines, extraction layout, and hash manifests are unchanged, so
//! an extraction the shell implementation published still verifies as
//! cached and is never re-extracted just because the tool changed.
//!
//! Run `cargo xtask bootstrap-native-dependencies` first so the verified
//! archives exist in the cache.

mod archive_extraction;
mod command;
mod hash_manifest;
mod header_locations;
mod options;
mod patch_pipeline;
mod pinned_archives;
mod provision_flow;
mod step_log;
mod verify_flow;

#[cfg(test)]
#[path = "bindgen_headers_tests.rs"]
mod bindgen_headers_tests;

pub use command::run;
