//! Typed current-process disk input/output evidence from macOS `proc_pid_rusage`.
//!
//! The counters are process-attributed physical disk input/output according to
//! XNU accounting. They are wider than expert paging: any worker disk activity
//! between two snapshots can contribute to a delta.

// The workspace confines unsafe code to runtime-integration's MLX-C boundary,
// and this module is model-serving's single documented exception, mirroring the
// daemon's system-telemetry module: sampling evidence is inseparable from
// the accounting syscall and C layout that produce it, and splitting the
// snapshot from its syscall would put an FFI seam through the evidence path.
// The unsafe surface never escapes this file — one synchronous call into a
// uniquely writable local buffer.
#![allow(unsafe_code)]

use thiserror::Error;

/// One cumulative process input/output sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MacosProcessIoSnapshot {
    physical_disk_read_bytes: u64,
    physical_disk_written_bytes: u64,
}

impl MacosProcessIoSnapshot {
    #[must_use]
    #[doc(hidden)]
    pub const fn from_cumulative_bytes(
        physical_disk_read_bytes: u64,
        physical_disk_written_bytes: u64,
    ) -> Self {
        Self {
            physical_disk_read_bytes,
            physical_disk_written_bytes,
        }
    }

    #[must_use]
    pub const fn physical_disk_read_bytes(self) -> u64 {
        self.physical_disk_read_bytes
    }

    #[must_use]
    pub const fn physical_disk_written_bytes(self) -> u64 {
        self.physical_disk_written_bytes
    }

    /// Computes one request/report interval from two cumulative process samples.
    ///
    /// A process restart, kernel anomaly, or future accounting reset can make a
    /// later counter smaller. Such a sample is unavailable evidence, not zero
    /// traffic and not unsigned wraparound, so subtraction fails explicitly.
    pub fn delta_since(
        self,
        earlier_snapshot: Self,
    ) -> Result<MacosProcessIoDelta, MacosProcessIoError> {
        let physical_disk_read_bytes = self
            .physical_disk_read_bytes
            .checked_sub(earlier_snapshot.physical_disk_read_bytes)
            .ok_or(MacosProcessIoError::CounterRegressed {
                counter_name: "ri_diskio_bytesread",
                earlier_bytes: earlier_snapshot.physical_disk_read_bytes,
                later_bytes: self.physical_disk_read_bytes,
            })?;
        let physical_disk_written_bytes = self
            .physical_disk_written_bytes
            .checked_sub(earlier_snapshot.physical_disk_written_bytes)
            .ok_or(MacosProcessIoError::CounterRegressed {
                counter_name: "ri_diskio_byteswritten",
                earlier_bytes: earlier_snapshot.physical_disk_written_bytes,
                later_bytes: self.physical_disk_written_bytes,
            })?;
        Ok(MacosProcessIoDelta {
            physical_disk_read_bytes,
            physical_disk_written_bytes,
        })
    }
}

/// Process-attributed physical disk input/output during one measured interval.
///
/// “Process-attributed” is deliberate: these values are not scoped to expert
/// files. Tokenizer, prompt-cache, logging, or unrelated worker reads during the
/// same report interval may contribute. They are still the correct companion to
/// logical `pread` bytes because they reveal whether macOS needed physical I/O.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MacosProcessIoDelta {
    physical_disk_read_bytes: u64,
    physical_disk_written_bytes: u64,
}

impl MacosProcessIoDelta {
    #[must_use]
    pub const fn physical_disk_read_bytes(self) -> u64 {
        self.physical_disk_read_bytes
    }

    #[must_use]
    pub const fn physical_disk_written_bytes(self) -> u64 {
        self.physical_disk_written_bytes
    }
}

/// A process input/output sample cannot be used as evidence.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum MacosProcessIoError {
    #[error("macOS proc_pid_rusage failed with operating-system error {os_error_code}")]
    SamplingFailed { os_error_code: i32 },
    #[error(
        "macOS process I/O counter {counter_name} regressed from {earlier_bytes} to {later_bytes} bytes"
    )]
    CounterRegressed {
        counter_name: &'static str,
        earlier_bytes: u64,
        later_bytes: u64,
    },
    #[error("process I/O accounting is available only on macOS")]
    UnsupportedPlatform,
}

/// One cumulative sampling attempt: the snapshot on success, the reason the
/// evidence is unavailable on failure. Consumers keep the `Err` rather than
/// substituting zero, so "unknown" stays distinguishable from "no traffic".
/// Only the direct-MLX streaming pager carries a sampling baseline, so the
/// alias is gated with it and a default build sees no dead alias.
#[cfg(feature = "direct-mlx")]
pub type MacosProcessIoSamplingOutcome = Result<MacosProcessIoSnapshot, MacosProcessIoError>;

/// Samples cumulative physical disk input/output attributed to this process.
///
/// This function performs no file I/O itself. The disabled performance path
/// never calls it; enabled attribution samples only at report boundaries.
pub fn sample_current_process_io() -> Result<MacosProcessIoSnapshot, MacosProcessIoError> {
    sample_process_io_for_platform()
}

/// Samples cumulative physical disk input/output attributed to another
/// same-user process, such as a supervised inference worker.
///
/// The kernel does not attribute every byte it moves on a process's behalf to
/// that process's own counters; external sampling of the child complements the
/// worker's self-sampling and makes the divergence itself measurable.
pub fn sample_process_io_for_process_id(
    process_id: u32,
) -> Result<MacosProcessIoSnapshot, MacosProcessIoError> {
    sample_process_io_for_process_id_on_platform(process_id)
}

#[cfg(target_os = "macos")]
fn sample_process_io_for_platform() -> Result<MacosProcessIoSnapshot, MacosProcessIoError> {
    // SAFETY: `getpid` only reads the caller's own process identity.
    let self_process_id = u32::try_from(unsafe { libc::getpid() }).map_err(|_| {
        MacosProcessIoError::SamplingFailed {
            os_error_code: libc::EINVAL,
        }
    })?;
    sample_process_io_for_process_id_on_platform(self_process_id)
}

#[cfg(target_os = "macos")]
fn sample_process_io_for_process_id_on_platform(
    process_id: u32,
) -> Result<MacosProcessIoSnapshot, MacosProcessIoError> {
    // A process id wider than the syscall's c_int cannot exist on the system,
    // so it is reported as a sampling failure rather than truncated.
    let process_id =
        i32::try_from(process_id).map_err(|_| MacosProcessIoError::SamplingFailed {
            os_error_code: libc::EINVAL,
        })?;
    // The layout comes from `libc`, never a local transcription: the kernel
    // writes the flavor's complete structure, and a hand copy that once
    // lacked the trailing `ri_runnable_time` let that write run eight bytes
    // past the buffer onto a saved register, crashing the worker on its first
    // streamed expert page. Version 4 is the first stable layout with the
    // cumulative disk counters; CURRENT would tie the layout to the build SDK.
    let mut resource_usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: `resource_usage` is `libc`'s `rusage_info_v4`, the exact layout
    // the V4 flavor fills, and is uniquely writable for this synchronous call.
    let status = unsafe {
        libc::proc_pid_rusage(
            process_id,
            libc::RUSAGE_INFO_V4,
            resource_usage.as_mut_ptr().cast(),
        )
    };
    if status != 0 {
        return Err(MacosProcessIoError::SamplingFailed {
            os_error_code: std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(status),
        });
    }
    // SAFETY: the buffer started zeroed and every field is a plain integer,
    // so it is initialised whether or not the kernel filled it.
    let resource_usage = unsafe { resource_usage.assume_init() };
    Ok(MacosProcessIoSnapshot::from_cumulative_bytes(
        resource_usage.ri_diskio_bytesread,
        resource_usage.ri_diskio_byteswritten,
    ))
}

#[cfg(not(target_os = "macos"))]
fn sample_process_io_for_platform() -> Result<MacosProcessIoSnapshot, MacosProcessIoError> {
    Err(MacosProcessIoError::UnsupportedPlatform)
}

#[cfg(not(target_os = "macos"))]
fn sample_process_io_for_process_id_on_platform(
    _process_id: u32,
) -> Result<MacosProcessIoSnapshot, MacosProcessIoError> {
    Err(MacosProcessIoError::UnsupportedPlatform)
}

#[cfg(test)]
mod tests;
