//! The FLUX.2 Klein render steps behind the shared image lifecycle. Loading,
//! finalizing, memory telemetry and live limits are the shared
//! [`ImageEngineComponents`]; this trait adds the phases FLUX runs.

use ipc_protocol::RequestId;

use super::super::{Flux2KleinFlowSchedule, Flux2KleinFlowStep, Flux2KleinImageDimensions};
use crate::image_engine_shell::{ImageComponentLoad, ImageEngineComponents};

/// Loaded identity supplied by either native components or a scripted lifecycle fake.
#[doc(hidden)]
pub type Flux2KleinComponentLoad = ImageComponentLoad;

/// Component seam keeps lifecycle tests scripted while production retains one concrete owner.
/// It deliberately has no `Send` bound because MLX component state is runtime-thread-affine.
#[doc(hidden)]
pub trait Flux2KleinEngineComponents: ImageEngineComponents {
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
}
