use astronomical_ipc_protocol::{ImageGenerationCommand, ImageGenerationFailureReason, RequestId};

use crate::{
    Flux2KleinImageEngine, ImageGenerationEngine, ImageGenerationEngineLoadResult,
    ImageGenerationEngineStep, MlxMemoryLimitAdjustment, MlxMemoryTelemetry,
    QwenImage21ImageEngine,
};

/// Family-tagged image engine used by the generic worker so one process can serve any
/// installed image-generation family.
pub enum ModelFamilyImageEngine {
    Flux2Klein(Flux2KleinImageEngine),
    QwenImage21(QwenImage21ImageEngine),
}

impl ImageGenerationEngine for ModelFamilyImageEngine {
    fn load(&mut self) -> Result<ImageGenerationEngineLoadResult, ImageGenerationFailureReason> {
        match self {
            Self::Flux2Klein(engine) => engine.load(),
            Self::QwenImage21(engine) => engine.load(),
        }
    }

    fn start_generation(
        &mut self,
        generation_command: ImageGenerationCommand,
    ) -> Result<(), ImageGenerationFailureReason> {
        match self {
            Self::Flux2Klein(engine) => engine.start_generation(generation_command),
            Self::QwenImage21(engine) => engine.start_generation(generation_command),
        }
    }

    fn advance_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason> {
        match self {
            Self::Flux2Klein(engine) => engine.advance_generation(request_id),
            Self::QwenImage21(engine) => engine.advance_generation(request_id),
        }
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<(), ImageGenerationFailureReason> {
        match self {
            Self::Flux2Klein(engine) => engine.cancel_generation(request_id),
            Self::QwenImage21(engine) => engine.cancel_generation(request_id),
        }
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        match self {
            Self::Flux2Klein(engine) => engine.take_post_cleanup_memory_telemetry(),
            Self::QwenImage21(engine) => engine.take_post_cleanup_memory_telemetry(),
        }
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        match self {
            Self::Flux2Klein(engine) => engine.collect_mlx_memory_telemetry(),
            Self::QwenImage21(engine) => engine.collect_mlx_memory_telemetry(),
        }
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, ImageGenerationFailureReason> {
        match self {
            Self::Flux2Klein(engine) => {
                engine.update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
            }
            Self::QwenImage21(engine) => {
                engine.update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
            }
        }
    }
}
