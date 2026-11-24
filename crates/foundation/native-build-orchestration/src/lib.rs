//! Orchestration for Astronomical's pinned native runtime: the CMake
//! invocation, the identity-keyed machine-wide build store, linker
//! directives, native parallelism resolution, and the hosted-CI progress
//! stream.
//!
//! # One compilation home, three consumers
//!
//! This crate is the single home of the build-orchestration sources.
//! `runtime-integration` consumes it from its build script through
//! `[build-dependencies]`, the standalone `native-build-tool`
//! pre-warms the same store through a normal dependency, and each module's
//! tests live beside it as `#[cfg(test)]` submodules. Byte-identical
//! execution between the pre-warm tool and the in-Cargo build follows from
//! compiling one crate, rather than from sharing files across crate roots
//! (issue #1628).
//!
//! # Executors versus policy
//!
//! These modules are executors and are deliberately excluded from the
//! native build identity pathspecs: edits to progress logging, parallelism,
//! or store plumbing must not force a full CMake rebuild. The
//! identity-relevant policy input beside them, `src/native-build-store-schema-version`,
//! is fingerprinted through `SOURCE_IDENTITY_PATHSPECS` in
//! `native-identity`.

pub mod native_compile;
pub mod native_host_platform;
pub mod native_link_directives;
pub mod native_linking;
pub mod native_parallelism;
pub mod native_progress;
pub mod native_store;

// Root re-exports so consumers address the orchestration surface through
// the crate root; the module paths stay public for narrower imports.
pub use native_compile::{
    ASTRO_NATIVE_BUILD_CACHE_DEPENDENCIES_VARIABLE, NATIVE_BUILD_STATUS_FILE_VARIABLE,
    NATIVE_BUILD_STORE_VARIABLE, NativeRuntimeBuildInputs, RUSTC_WRAPPER_VARIABLE,
    build_pinned_native_runtime, detect_host_target_triple, native_build_store_directory,
    native_parallel_job_count, required_path_variable, resolve_native_build_identity,
    write_native_build_status,
};
pub use native_host_platform::{
    NativeTargetPlatform, identity_environment_variables, native_build_platform,
};
pub use native_linking::configure_rust_linking;
pub use native_progress::{NATIVE_BUILD_PROGRESS_FILE_VARIABLE, NativeBuildProgress};
pub use native_store::{NativeBuildProfile, NativeBuildStore};
