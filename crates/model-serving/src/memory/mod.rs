mod adaptive_ram_growth_guard;
mod mlx_memory_telemetry;

pub use adaptive_ram_growth_guard::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, AdaptiveRamGrowthGuardError,
    AdaptiveRamGrowthPhase, AdaptiveRamGrowthProjection,
};
pub use mlx_memory_telemetry::{
    MlxActiveMemoryBreakdown, MlxMemoryLimitAdjustment, MlxMemoryTelemetry,
};
