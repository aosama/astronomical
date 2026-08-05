use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

mod metal_expert_pack_loader;

fn runtime() -> MlxRuntime {
    let mlx_memory_limits = MlxMemoryLimits::new(2_000_000_000, 256_000_000)
        .expect("the experimental test memory limits should be valid");
    MlxRuntime::initialize(mlx_memory_limits)
        .expect("the pinned MLX runtime should initialize for the experimental contract")
}
