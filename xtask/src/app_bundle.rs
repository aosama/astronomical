//! `cargo xtask app` — builds the macOS app bundle from typed channel
//! profiles. `app dev` is the Development edit loop: persistent incremental
//! outputs, cached icon and npm install, ad-hoc signing, structural checks,
//! then a swap of the running app. `app instance` controls that instance
//! headless. Release packaging still runs through
//! `scripts/internal/build-macos-app.sh` until `app package` replaces it.

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
mod native_metallib;
mod running_instance;
mod step_reporter;
mod tool_command;

#[cfg(test)]
#[path = "app_bundle_tests.rs"]
mod app_bundle_tests;

pub use command::run;
