//! Maps each platform's memory-pressure reading onto the public telemetry
//! states `normal`, `warning` and `critical`. Pure text parsing, so every
//! platform's mapping is tested on every platform.

const MEMORY_PRESSURE_NORMAL_BIT: u32 = 1;
const MEMORY_PRESSURE_WARNING_BIT: u32 = 2;
const MEMORY_PRESSURE_CRITICAL_BIT: u32 = 4;
const MEMORY_PRESSURE_KNOWN_BITS: u32 =
    MEMORY_PRESSURE_NORMAL_BIT | MEMORY_PRESSURE_WARNING_BIT | MEMORY_PRESSURE_CRITICAL_BIT;

/// The share of the last ten seconds, in percent, that at least one task
/// (`some`) or every task (`full`) spent stalled on memory, at or past which
/// Linux pressure stall information reads as a warning or as critical. Ten
/// percent of wall time lost to memory stalls is when reclaim starts to be
/// felt. Every task stalled for a tenth of that time means the whole system
/// waits on memory, the closest match to macOS's critical level; a `full`
/// stall below that threshold, even a sustained one, reads by its `some`
/// share instead.
const LINUX_WARNING_SOME_STALL_PERCENTAGE: f64 = 10.0;
const LINUX_CRITICAL_FULL_STALL_PERCENTAGE: f64 = 10.0;

/// Maps the macOS memory-pressure sysctl bitmask to the public telemetry state.
/// Unknown and empty values remain unavailable instead of becoming Normal.
pub fn parse_macos_memory_pressure_level(sysctl_value_text: &str) -> Option<&'static str> {
    let memory_pressure_bitmask = sysctl_value_text.trim().parse::<u32>().ok()?;
    if memory_pressure_bitmask == 0 || memory_pressure_bitmask & !MEMORY_PRESSURE_KNOWN_BITS != 0 {
        return None;
    }
    if memory_pressure_bitmask & MEMORY_PRESSURE_CRITICAL_BIT != 0 {
        return Some("critical");
    }
    if memory_pressure_bitmask & MEMORY_PRESSURE_WARNING_BIT != 0 {
        return Some("warning");
    }
    if memory_pressure_bitmask & MEMORY_PRESSURE_NORMAL_BIT != 0 {
        return Some("normal");
    }
    None
}

/// Maps Linux's pressure stall information (`/proc/pressure/memory`) onto the
/// public telemetry state, from the ten-second averages. A file without both
/// lines, or a kernel built without pressure stall information, reads as
/// unavailable instead of becoming Normal.
pub fn parse_linux_memory_pressure_level(pressure_stall_text: &str) -> Option<&'static str> {
    let some_stall_percentage = ten_second_stall_percentage(pressure_stall_text, "some")?;
    let full_stall_percentage = ten_second_stall_percentage(pressure_stall_text, "full")?;
    if full_stall_percentage >= LINUX_CRITICAL_FULL_STALL_PERCENTAGE {
        Some("critical")
    } else if some_stall_percentage >= LINUX_WARNING_SOME_STALL_PERCENTAGE {
        Some("warning")
    } else {
        Some("normal")
    }
}

/// Reads `avg10=` from the line that starts with `stall_kind`, such as
/// `some avg10=1.25 avg60=0.40 avg300=0.10 total=123`.
fn ten_second_stall_percentage(pressure_stall_text: &str, stall_kind: &str) -> Option<f64> {
    for pressure_line in pressure_stall_text.lines() {
        let mut line_fields = pressure_line.split_whitespace();
        if line_fields.next() != Some(stall_kind) {
            continue;
        }
        for line_field in line_fields {
            if let Some(average_text) = line_field.strip_prefix("avg10=") {
                let stall_percentage = average_text.parse::<f64>().ok()?;
                let is_valid_percentage = (0.0..=100.0).contains(&stall_percentage);
                return if is_valid_percentage {
                    Some(stall_percentage)
                } else {
                    None
                };
            }
        }
        return None;
    }
    None
}

#[cfg(test)]
#[path = "memory_pressure_levels_tests.rs"]
mod tests;
