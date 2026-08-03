/// Resolves the sampling seed for one generation request.
///
/// When the client supplies an explicit seed, it is used directly for deterministic
/// generation. When the client omits the seed (sends `null`), Astronomical uses the
/// current time in milliseconds since the Unix epoch, matching MLX's default
/// `PyKeySequence` behavior in `mlx/python/src/random.cpp`.
///
/// The previous behavior used `request_id` as the fallback seed, which made every
/// request deterministic to its request identifier. Under that scheme, if one
/// request produced a hallucinated tool name, retrying the same request_id would
/// produce the same wrong output every time. Using a time-based seed gives each
/// request fresh sampling entropy when
/// the client does not request determinism.
pub fn resolve_sampling_seed(client_seed: Option<u64>, time_provider: impl Fn() -> u64) -> u64 {
    client_seed.unwrap_or_else(&time_provider)
}

/// Returns the current time in milliseconds since the Unix epoch.
///
/// This is the production time provider passed to [`resolve_sampling_seed`].
/// It matches MLX's `std::chrono::system_clock` seed source.
#[cfg(feature = "direct-mlx")]
#[must_use]
pub fn current_time_millis_since_unix_epoch() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}
