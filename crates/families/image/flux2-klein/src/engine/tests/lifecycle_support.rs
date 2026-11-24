//! Scripted FLUX component owner shared by lifecycle user-journey tests.

use std::sync::{Arc, Mutex};

use crate::{
    FLUX2_KLEIN_OFFICIAL_MODEL_ID, FLUX2_KLEIN_OFFICIAL_REVISION, Flux2KleinComponentLoad,
    Flux2KleinEngineComponents, Flux2KleinFlowStep, Flux2KleinImageDimensions,
    Flux2KleinImageEngine,
};
use image_generation::image_engine_shell::ImageEngineComponents;
pub(super) use image_generation::image_engine_shell_test_support::{
    assert_final_cleanup_memory, cloned_events,
};
use image_generation::image_engine_shell_test_support::{
    finalize_event, record_lifecycle_event, scripted_post_cleanup_memory,
};
use ipc_protocol::{
    ImageGenerationCapabilities, ImageGenerationCommand, ImageGenerationSettings, RequestId,
};
use memory_policy::telemetry::MlxMemoryTelemetry;
use model_primitives::performance_attribution::PerformanceAttributionOutcome;

pub(super) fn fake_engine(
    lifecycle_events: Arc<Mutex<Vec<String>>>,
    failing_event: Option<&str>,
) -> Flux2KleinImageEngine {
    scripted_engine(lifecycle_events, failing_event, 0, 0)
}

pub(super) fn bounded_fake_engine(
    lifecycle_events: Arc<Mutex<Vec<String>>>,
    groups_before_euler: usize,
) -> Flux2KleinImageEngine {
    scripted_engine(lifecycle_events, None, groups_before_euler, 0)
}

pub(super) fn bounded_conditioning_fake_engine(
    lifecycle_events: Arc<Mutex<Vec<String>>>,
    conditioning_groups_before_complete: usize,
) -> Flux2KleinImageEngine {
    scripted_engine(
        lifecycle_events,
        None,
        0,
        conditioning_groups_before_complete,
    )
}

fn scripted_engine(
    lifecycle_events: Arc<Mutex<Vec<String>>>,
    failing_event: Option<&str>,
    groups_before_euler: usize,
    conditioning_groups_before_complete: usize,
) -> Flux2KleinImageEngine {
    Flux2KleinImageEngine::from_components(
        FLUX2_KLEIN_OFFICIAL_MODEL_ID,
        Box::new(FakeComponents {
            lifecycle_events,
            failing_event: failing_event.map(str::to_owned),
            groups_before_euler,
            next_group_index: 0,
            conditioning_groups_before_complete,
            next_conditioning_group_index: 0,
            post_cleanup_memory_telemetry: None,
        }),
    )
}

pub(super) fn valid_command(request_id: u64, seed: u64) -> ImageGenerationCommand {
    ImageGenerationCommand {
        request_id: RequestId::new(request_id),
        model: FLUX2_KLEIN_OFFICIAL_MODEL_ID.to_owned(),
        prompt: "Romeo and Juliet".to_owned(),
        settings: ImageGenerationSettings {
            width_pixels: 64,
            height_pixels: 64,
            steps: 4,
            guidance_thousandths: 1_000,
            seed,
        },
    }
}

struct FakeComponents {
    lifecycle_events: Arc<Mutex<Vec<String>>>,
    failing_event: Option<String>,
    groups_before_euler: usize,
    next_group_index: usize,
    conditioning_groups_before_complete: usize,
    next_conditioning_group_index: usize,
    post_cleanup_memory_telemetry: Option<MlxMemoryTelemetry>,
}

impl ImageEngineComponents for FakeComponents {
    fn load(&mut self) -> Result<Flux2KleinComponentLoad, String> {
        self.record("load")?;
        Ok(Flux2KleinComponentLoad::new(
            FLUX2_KLEIN_OFFICIAL_MODEL_ID,
            FLUX2_KLEIN_OFFICIAL_REVISION,
            ImageGenerationCapabilities {
                minimum_width_pixels: 64,
                maximum_width_pixels: 1_024,
                minimum_height_pixels: 64,
                maximum_height_pixels: 1_024,
                dimension_multiple_pixels: 16,
                maximum_steps: 4,
                maximum_guidance_thousandths: 1_000,
                output_mime_types: vec!["image/png".to_owned()],
            },
            400_000_000,
        ))
    }

    fn finalize_request(
        &mut self,
        outcome: PerformanceAttributionOutcome,
        encoded_bytes: Option<u64>,
        failure_description: Option<&str>,
    ) -> Result<(), String> {
        self.record(&finalize_event(outcome, encoded_bytes, failure_description))?;
        self.post_cleanup_memory_telemetry = Some(scripted_post_cleanup_memory());
        Ok(())
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.post_cleanup_memory_telemetry.take()
    }
}

impl Flux2KleinEngineComponents for FakeComponents {
    fn start_request(
        &mut self,
        _request_id: RequestId,
        dimensions: Flux2KleinImageDimensions,
        seed: u64,
    ) -> Result<crate::scheduler::Flux2KleinFlowSchedule, String> {
        assert_eq!(
            (dimensions.width_pixels(), dimensions.height_pixels()),
            (64, 64)
        );
        self.next_group_index = 0;
        self.next_conditioning_group_index = 0;
        self.post_cleanup_memory_telemetry = None;
        self.record(&format!("start:{seed}"))?;
        crate::scheduler::Flux2KleinFlowScheduler::schedule(
            dimensions.width_pixels(),
            dimensions.height_pixels(),
        )
        .map_err(|error| error.to_string())
    }

    fn condition_prompt(&mut self, prompt: &str) -> Result<bool, String> {
        if self.conditioning_groups_before_complete == 0 {
            self.record(&format!("condition:{prompt}"))?;
            return Ok(true);
        }
        if self.next_conditioning_group_index < self.conditioning_groups_before_complete {
            self.record(&format!(
                "condition-layer:{}",
                self.next_conditioning_group_index
            ))?;
            self.next_conditioning_group_index += 1;
            return Ok(false);
        }
        self.record("condition-complete")?;
        Ok(true)
    }

    fn initialize_keyed_noise(&mut self, seed: u64, _initial_sigma: f64) -> Result<(), String> {
        self.record(&format!("noise:{seed}"))
    }

    fn denoise_euler(
        &mut self,
        step_index: usize,
        _flow_step: Flux2KleinFlowStep,
    ) -> Result<bool, String> {
        if self.next_group_index < self.groups_before_euler {
            self.record(&format!("group:{step_index}:{}", self.next_group_index))?;
            self.next_group_index += 1;
            return Ok(false);
        }
        self.next_group_index = 0;
        self.record(&format!("denoise:{step_index}"))?;
        Ok(true)
    }

    fn decode_latents(&mut self) -> Result<bool, String> {
        self.record("decode")?;
        Ok(true)
    }

    fn encode_png(&mut self) -> Result<Vec<u8>, String> {
        self.record("encode")?;
        Ok(b"complete png".to_vec())
    }
}

impl FakeComponents {
    fn record(&self, event: &str) -> Result<(), String> {
        record_lifecycle_event(&self.lifecycle_events, self.failing_event.as_deref(), event)
    }
}
