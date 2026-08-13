//! Typed current-process disk input/output evidence from macOS `proc_pid_rusage`.
//!
//! The counters are process-attributed physical disk input/output according to
//! XNU accounting. They are wider than expert paging: any worker disk activity
//! between two snapshots can contribute to a delta.

use thiserror::Error;

// Keep the flavor and Rust layout together. Version 4 is old enough for every
// supported Astronomical macOS deployment and is the first stable layout that
// contains the cumulative disk read/write fields needed here. Using CURRENT
// would make the FFI layout silently depend on the build SDK.
const RUSAGE_INFO_V4: i32 = 4;

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

/// Samples cumulative physical disk input/output attributed to this process.
///
/// This function performs no file I/O itself. The disabled performance path
/// never calls it; enabled attribution samples only at report boundaries.
pub fn sample_current_process_io() -> Result<MacosProcessIoSnapshot, MacosProcessIoError> {
    sample_current_process_io_for_platform()
}

#[cfg(target_os = "macos")]
fn sample_current_process_io_for_platform() -> Result<MacosProcessIoSnapshot, MacosProcessIoError> {
    let mut resource_usage = MacosResourceUsageInfoV4::default();
    // SAFETY: `resource_usage` has the SDK's `rusage_info_v4` C layout and is
    // uniquely writable for this synchronous call. `getpid` returns this process.
    let status = unsafe {
        proc_pid_rusage(
            libc::getpid(),
            RUSAGE_INFO_V4,
            (&raw mut resource_usage).cast(),
        )
    };
    if status != 0 {
        return Err(MacosProcessIoError::SamplingFailed {
            os_error_code: std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(status),
        });
    }
    Ok(MacosProcessIoSnapshot::from_cumulative_bytes(
        resource_usage.ri_diskio_bytesread,
        resource_usage.ri_diskio_byteswritten,
    ))
}

#[cfg(not(target_os = "macos"))]
fn sample_current_process_io_for_platform() -> Result<MacosProcessIoSnapshot, MacosProcessIoError> {
    Err(MacosProcessIoError::UnsupportedPlatform)
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Default)]
struct MacosResourceUsageInfoV4 {
    // This is an exact transcription of Apple's public `rusage_info_v4` layout.
    // Fields preceding disk I/O cannot be omitted because C writes the complete
    // structure. Fields following disk I/O remain present so the allocation is
    // large enough for the selected flavor on every supported SDK/runtime pair.
    ri_uuid: [u8; 16],
    ri_user_time: u64,
    ri_system_time: u64,
    ri_pkg_idle_wkups: u64,
    ri_interrupt_wkups: u64,
    ri_pageins: u64,
    ri_wired_size: u64,
    ri_resident_size: u64,
    ri_phys_footprint: u64,
    ri_proc_start_abstime: u64,
    ri_proc_exit_abstime: u64,
    ri_child_user_time: u64,
    ri_child_system_time: u64,
    ri_child_pkg_idle_wkups: u64,
    ri_child_interrupt_wkups: u64,
    ri_child_pageins: u64,
    ri_child_elapsed_abstime: u64,
    ri_diskio_bytesread: u64,
    ri_diskio_byteswritten: u64,
    ri_cpu_time_qos_default: u64,
    ri_cpu_time_qos_maintenance: u64,
    ri_cpu_time_qos_background: u64,
    ri_cpu_time_qos_utility: u64,
    ri_cpu_time_qos_legacy: u64,
    ri_cpu_time_qos_user_initiated: u64,
    ri_cpu_time_qos_user_interactive: u64,
    ri_billed_system_time: u64,
    ri_serviced_system_time: u64,
    ri_logical_writes: u64,
    ri_lifetime_max_phys_footprint: u64,
    ri_instructions: u64,
    ri_cycles: u64,
    ri_billed_energy: u64,
    ri_serviced_energy: u64,
    ri_interval_max_phys_footprint: u64,
}

#[cfg(target_os = "macos")]
#[link(name = "proc")]
unsafe extern "C" {
    fn proc_pid_rusage(process_id: i32, flavor: i32, buffer: *mut std::ffi::c_void) -> i32;
}
