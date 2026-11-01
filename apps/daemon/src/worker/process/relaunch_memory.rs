//! Launch-snapshot memory ceiling adopted after a live MLX memory update.
//!
//! `relaunch_after_termination` is the only replacement path that reuses the
//! snapshot captured when the process was spawned. A live ceiling raise updates
//! the running worker and the persisted configuration, then a later cancellation
//! failure starts a new process from that original snapshot. Recording the
//! applied ceiling here is what makes the replacement boot at the ceiling the
//! user already asked for.

use astronomical_ipc_protocol::WorkerStartupConfiguration;

/// Returns the launch snapshot a replacement process should boot from after a
/// live ceiling was adopted.
///
/// Stores the requested configured ceiling, not the wired-clamped engine
/// ceiling. Startup applies the wired-memory margin again, so storing the
/// already-clamped bytes would shrink the budget a second time.
///
/// A missing snapshot stays missing: this function cannot invent executable
/// paths or logging directories. A zero ceiling or a blank generation is
/// refused so a bad update cannot overwrite a snapshot that already starts.
#[must_use]
pub(crate) fn startup_configuration_carrying_applied_memory_ceiling(
    launch_startup_configuration: Option<WorkerStartupConfiguration>,
    configured_maximum_mlx_memory_bytes: u64,
    configuration_generation: &str,
) -> Option<WorkerStartupConfiguration> {
    if configured_maximum_mlx_memory_bytes == 0
        || configuration_generation_is_blank(configuration_generation)
    {
        return launch_startup_configuration;
    }
    let mut adopted_startup_configuration = launch_startup_configuration?;
    adopted_startup_configuration.configured_maximum_mlx_memory_bytes =
        Some(configured_maximum_mlx_memory_bytes);
    adopted_startup_configuration.configuration_generation = configuration_generation.to_owned();
    Some(adopted_startup_configuration)
}

/// An empty or all-whitespace generation cannot be acknowledged on relaunch.
///
/// Recovery compares the replacement process's acknowledged generation with
/// this snapshot. Writing a blank generation would make that comparison fail
/// for every later replacement.
fn configuration_generation_is_blank(configuration_generation: &str) -> bool {
    for character in configuration_generation.chars() {
        if !character.is_whitespace() {
            return false;
        }
    }
    true
}

#[cfg(test)]
#[path = "relaunch_memory_tests.rs"]
mod relaunch_memory_tests;
