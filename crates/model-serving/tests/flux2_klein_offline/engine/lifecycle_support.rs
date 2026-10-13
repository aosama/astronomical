//! Scripted FLUX component owner shared by lifecycle user-journey tests.

use std::sync::{Arc, Mutex};

use astronomical_ipc_protocol::{
    ImageGenerationCapabilities, ImageGenerationCommand, ImageGenerationSettings, RequestId,
};
use astronomical_model_serving::{
    FLUX2_KLEIN_OFFICIAL_MODEL_ID, FLUX2_KLEIN_OFFICIAL_REVISION, Flux2KleinComponentLoad,
    Flux2KleinEngineComponents, Flux2KleinFlowStep, Flux2KleinImageDimensions,
    Flux2KleinImageEngine, MlxActiveMemoryBreakdown, MlxMemoryTelemetry,
    PerformanceAttributionOutcome,
};

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
    Flux2KleinImageEngine::with_components_for_tests(
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

pub(super) fn cloned_events(lifecycle_events: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    lifecycle_events
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

pub(super) fn assert_final_cleanup_memory(memory_telemetry: Option<MlxMemoryTelemetry>) {
    let memory_telemetry =
        memory_telemetry.expect("finalized FLUX requests should expose MLX memory");
    assert_eq!(memory_telemetry.allocator_cache_memory_bytes, 0);
    assert!(memory_telemetry.active_memory_bytes <= memory_telemetry.peak_memory_bytes);
    assert_eq!(
        memory_telemetry.active_memory_breakdown,
        MlxActiveMemoryBreakdown::default()
    );
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

impl Flux2KleinEngineComponents for FakeComponents {
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

    fn start_request(
        &mut self,
        _request_id: RequestId,
        dimensions: Flux2KleinImageDimensions,
        seed: u64,
    ) -> Result<astronomical_model_serving::Flux2KleinFlowSchedule, String> {
        assert_eq!(
            (dimensions.width_pixels(), dimensions.height_pixels()),
            (64, 64)
        );
        self.next_group_index = 0;
        self.next_conditioning_group_index = 0;
        self.post_cleanup_memory_telemetry = None;
        self.record(&format!("start:{seed}"))?;
        astronomical_model_serving::Flux2KleinFlowScheduler::schedule(
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

    fn finalize_request(
        &mut self,
        outcome: PerformanceAttributionOutcome,
        encoded_bytes: Option<u64>,
        failure_description: Option<&str>,
    ) -> Result<(), String> {
        let outcome_name = match outcome {
            PerformanceAttributionOutcome::Success => "success",
            PerformanceAttributionOutcome::Rejected => "rejected",
            PerformanceAttributionOutcome::Cancelled => "cancelled",
            PerformanceAttributionOutcome::Failed => "failed",
        };
        let outcome_detail = failure_description
            .map(str::to_owned)
            .unwrap_or_else(|| encoded_bytes.unwrap_or(0).to_string());
        self.record(&format!("finalize:{outcome_name}:{outcome_detail}"))?;
        self.post_cleanup_memory_telemetry = Some(MlxMemoryTelemetry::new(
            96_000_000,
            0,
            512_000_000,
            MlxActiveMemoryBreakdown::default(),
        ));
        Ok(())
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.post_cleanup_memory_telemetry.take()
    }
}

impl FakeComponents {
    fn record(&self, event: &str) -> Result<(), String> {
        self.lifecycle_events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event.to_owned());
        if self.failing_event.as_deref() == Some(event) {
            Err(format!("injected failure at {event}"))
        } else {
            Ok(())
        }
    }
}
