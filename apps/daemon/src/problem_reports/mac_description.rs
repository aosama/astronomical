//! The Mac described in a report: its chip, memory and macOS version, read
//! once from `sysctl`, which needs no permission and no network.

use std::process::Command;

use serde::Serialize;

/// Bytes in one GB as Astronomical counts it, the gigabyte Apple uses for a
/// Mac's memory, so the report matches what the menu bar shows.
const BYTES_PER_GB: u64 = 1_073_741_824;

/// The three `sysctl` keys read, in the order their values are printed.
const SYSCTL_KEYS: [&str; 3] = [
    "machdep.cpu.brand_string",
    "hw.memsize",
    "kern.osproductversion",
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct MacDescription {
    pub(crate) chip_name: String,
    pub(crate) memory_gb: u64,
    pub(crate) macos_version: String,
}

impl MacDescription {
    /// Reads this Mac, or `None` when `sysctl` cannot answer, in which case
    /// no report is offered rather than one with made-up values.
    pub(crate) fn read_this_mac() -> Option<Self> {
        let sysctl_answer = Command::new("/usr/sbin/sysctl")
            .arg("-n")
            .args(SYSCTL_KEYS)
            .output()
            .ok()?;
        if !sysctl_answer.status.success() {
            return None;
        }
        Self::from_sysctl_values(&String::from_utf8_lossy(&sysctl_answer.stdout))
    }

    /// Parses the three values `sysctl -n` prints one per line: the chip
    /// name, the memory in bytes and the macOS version.
    pub(crate) fn from_sysctl_values(sysctl_values_text: &str) -> Option<Self> {
        let mut sysctl_lines = sysctl_values_text.lines();
        let chip_name = sysctl_lines.next()?.trim();
        let memory_bytes: u64 = sysctl_lines.next()?.trim().parse().ok()?;
        let macos_version = sysctl_lines.next()?.trim();
        if chip_name.is_empty() || macos_version.is_empty() || memory_bytes == 0 {
            return None;
        }
        Some(Self {
            chip_name: chip_name.to_owned(),
            memory_gb: rounded_memory_gb(memory_bytes),
            macos_version: macos_version.to_owned(),
        })
    }

    /// "Apple M5 Pro, 48 GB, macOS 27.2", the Mac line of a readable report.
    pub(crate) fn display_text(&self) -> String {
        format!(
            "{}, {} GB, macOS {}",
            self.chip_name, self.memory_gb, self.macos_version
        )
    }
}

/// Memory to the nearest whole GB; Macs ship in whole GB, so rounding only
/// removes the few megabytes some models report differently.
fn rounded_memory_gb(memory_bytes: u64) -> u64 {
    (memory_bytes + BYTES_PER_GB / 2) / BYTES_PER_GB
}

#[cfg(test)]
#[path = "mac_description_tests.rs"]
mod mac_description_tests;
