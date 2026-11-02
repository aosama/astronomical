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
    /// so `m3 max` wins over `m3`. Matching is a case-insensitive substring
    /// test with `_` in a key read as a space, as OptiQ spells its keys
    /// (`m3_max`). An unknown device gets [`Self::conservative_fallback`].
    #[must_use]
    pub fn from_device_name(device_name: &str, known_profiles: &[(&str, HardwareProfile)]) -> Self {
        let lowercase_device_name = device_name.to_lowercase();
        let mut best_match: Option<(usize, &HardwareProfile)> = None;
        for (profile_key, profile) in known_profiles {
            let spaced_key = profile_key.to_lowercase().replace('_', " ");
            let is_longer_than_best = match best_match {
                Some((best_key_length, _)) => spaced_key.len() > best_key_length,
                None => true,
            };
            if is_longer_than_best && lowercase_device_name.contains(&spaced_key) {
                best_match = Some((spaced_key.len(), profile));
            }
        }
        match best_match {
            Some((_, profile)) => profile.clone(),
            None => Self::conservative_fallback(device_name),
        }
    }
}
