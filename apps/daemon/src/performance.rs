//! Performance attribution and measurement logs.

pub(crate) mod completion_attribution_log;
pub(crate) mod daemon_download_attribution;
pub(crate) mod daemon_performance_attribution;
pub(crate) mod daemon_performance_measurement;
pub(crate) mod daemon_performance_record;
pub(crate) mod generation_performance_log;

#[cfg(test)]
mod completion_attribution_log_tests;
#[cfg(test)]
mod daemon_performance_attribution_tests;
#[cfg(test)]
mod generation_performance_log_tests;
