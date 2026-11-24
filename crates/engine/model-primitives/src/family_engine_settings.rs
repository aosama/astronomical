//! What the worker hands every model family's engine builder (issue #2473,
//! Step 5): the memory limits and the performance-recording settings that
//! are the same whichever family is served. It lives here because this is
//! the one crate every family builds on.

use std::path::PathBuf;

/// The worker settings every family's engine builder reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyEngineSettings {
    /// The MLX active-memory ceiling the engine plans its loads under.
    pub effective_mlx_memory_ceiling_bytes: usize,
    /// The ceiling of the `mlx_buffer_cache` the engine keeps.
    pub mlx_buffer_cache_limit_bytes: usize,
    /// The highest ceiling an image engine may plan its load at to reach its
    /// floor (#1902): the user's own limit, or unbounded when they set none.
    pub floor_raise_limit_bytes: u64,
    /// Whether the engine records where each request's time went.
    pub performance_attribution_enabled: bool,
    /// Where those records are written.
    pub performance_attribution_log_path: PathBuf,
}
