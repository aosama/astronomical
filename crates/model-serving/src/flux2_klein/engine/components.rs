//! Architecture-neutral component contract behind the FLUX lifecycle owner.

use astronomical_ipc_protocol::{ImageGenerationCapabilities, RequestId};

use crate::{MlxMemoryLimitAdjustment, MlxMemoryTelemetry, PerformanceAttributionOutcome};

use super::super::{Flux2KleinFlowSchedule, Flux2KleinFlowStep, Flux2KleinImageDimensions};

/// Loaded identity supplied by either native components or a hermetic lifecycle fake.
#[doc(hidden)]
pub struct Flux2KleinComponentLoad {
    pub(super) model_id: String,
    pub(super) revision: String,
    pub(super) capabilities: ImageGenerationCapabilities,
    pub(super) minimum_mlx_memory_ceiling_bytes: u64,
}

impl Flux2KleinComponentLoad {
    #[doc(hidden)]
    pub fn new(
        model_id: impl Into<String>,
        revision: impl Into<String>,
        capabilities: ImageGenerationCapabilities,
        minimum_mlx_memory_ceiling_bytes: u64,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            revision: revision.into(),
            capabilities,
            minimum_mlx_memory_ceiling_bytes,
        }
    }
}

/// Component seam keeps lifecycle tests hermetic while production retains one concrete owner.
/// It deliberately has no `Send` bound because MLX component state is runtime-thread-affine.
#[doc(hidden)]
pub trait Flux2KleinEngineComponents {
    fn load(&mut self) -> Result<Flux2KleinComponentLoad, String>;
    fn start_request(
        &mut self,
        request_id: RequestId,
        dimensions: Flux2KleinImageDimensions,
        seed: u64,
    ) -> Result<Flux2KleinFlowSchedule, String>;
    /// Returns true only after all text-encoder layers and tap concatenation complete.
    fn condition_prompt(&mut self, prompt: &str) -> Result<bool, String>;
    fn initialize_keyed_noise(&mut self, seed: u64, initial_sigma: f64) -> Result<(), String>;
    /// Returns true only after the final group output and Euler update are evaluated.
    fn denoise_euler(
        &mut self,
        step_index: usize,
        flow_step: Flux2KleinFlowStep,
    ) -> Result<bool, String>;
    /// Returns true only when the complete decoded pixel array is ready for encoding.
    fn decode_latents(&mut self) -> Result<bool, String>;
    fn encode_png(&mut self) -> Result<Vec<u8>, String>;
    fn finalize_request(
        &mut self,
        outcome: PerformanceAttributionOutcome,
        encoded_bytes: Option<u64>,
        failure_description: Option<&str>,
    ) -> Result<(), String>;
    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        None
    }
    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        None
    }
    fn update_mlx_memory_limit(
        &mut self,
        _requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, String> {
        Err("live MLX memory limits are unavailable".to_owned())
    }
}
