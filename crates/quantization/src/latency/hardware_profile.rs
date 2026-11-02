//! Apple Silicon GPU characteristics the roofline model reads: memory
//! bandwidth and peak arithmetic throughput per device family.

/// Apple Silicon GPU characteristics.
#[derive(Clone, Debug, PartialEq)]
pub struct HardwareProfile {
    pub name: String,
    pub memory_bandwidth_bytes_per_second: f64,
    pub peak_float16_operations_per_second: f64,
    pub gpu_core_count: u32,
}

impl HardwareProfile {
    /// A conservative generic profile for an unrecognized Apple Silicon
    /// device, OptiQ's fallback (200 GB/s, 7 TFLOPS, 16 cores).
    #[must_use]
    pub fn conservative_fallback(device_name: &str) -> Self {
        Self {
            name: device_name.to_owned(),
            memory_bandwidth_bytes_per_second: 200.0e9,
            peak_float16_operations_per_second: 7.0e12,
            gpu_core_count: 16,
        }
    }

    /// Looks a device name up in a caller-supplied table, longest key first
    /// so `m3 max` wins over `m3`.
    #[must_use]
    pub fn from_device_name(device_name: &str, known_profiles: &[(&str, HardwareProfile)]) -> Self {
        let _ = (device_name, known_profiles);
        todo!("case-insensitive substring match, longest key first, else conservative_fallback")
    }
}
