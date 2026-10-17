//! Injected-stage acceptance coverage for bounded VAE advancement and cancellation.

use std::sync::{Arc, Mutex};

use astronomical_ipc_protocol::{
    ImageGenerationCapabilities, ImageGenerationCommand, ImageGenerationPhase,
    ImageGenerationSettings, RequestId,
};
use astronomical_model_serving::{
    FLUX2_KLEIN_OFFICIAL_MODEL_ID, FLUX2_KLEIN_OFFICIAL_REVISION, Flux2KleinComponentLoad,
    Flux2KleinEngineComponents, Flux2KleinFlowSchedule, Flux2KleinFlowScheduler,
    Flux2KleinFlowStep, Flux2KleinImageDimensions, Flux2KleinImageEngine, ImageGenerationEngine,
    ImageGenerationEngineStep, PerformanceAttributionOutcome,
};

#[test]
fn should_advance_one_complete_decoder_stage_per_call() {
    let complete_events = Arc::new(Mutex::new(Vec::new()));
    let complete_pixels = Arc::new(Mutex::new(Vec::new()));
    let complete_png = run_to_completion(
        staged_engine(
            Arc::clone(&complete_events),
            Arc::clone(&complete_pixels),
            &[
                "complete:input",
                "complete:middle",
                "complete:up-0",
                "complete:output",
            ],
        ),
        61,
    );

    assert!(!complete_png.is_empty());
    assert!(!cloned_pixels(&complete_pixels).is_empty());
    assert_eq!(
        decode_events(&complete_events),
        vec![
            "decode:complete:input",
            "decode:complete:middle",
            "decode:complete:up-0",
            "decode:complete:output",
        ]
    );
}

#[test]
fn should_cancel_between_complete_decode_stages_cleanup_before_acknowledgement_and_reuse_request_state()
 {
    let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
    let decoded_pixels = Arc::new(Mutex::new(Vec::new()));
    let mut engine = staged_engine(
        Arc::clone(&lifecycle_events),
        decoded_pixels,
        &[
            "complete:input",
            "complete:middle-before-attention",
            "complete:middle-attention",
            "complete:middle-after-attention",
        ],
    );
    engine.load().expect("the staged component should load");
    engine
        .start_generation(valid_command(63))
        .expect("the first request should start");
    advance_to_decoding(&mut engine, 63);

    for _bounded_stage_index in 0..3 {
        let ImageGenerationEngineStep::Progress {
            phase,
            completed_steps,
            ..
        } = engine
            .advance_generation(RequestId::new(63))
            .expect("one complete decoder stage should advance")
        else {
            panic!("partial complete VAE work must remain progress");
        };
        assert_eq!(phase, ImageGenerationPhase::Decoding);
        assert_eq!(completed_steps, 4);
    }
    engine
        .cancel_generation(RequestId::new(63))
        .expect("cancellation should release and synchronize VAE state before acknowledgement");

    let events_after_cancellation = cloned_events(&lifecycle_events);
    assert!(events_after_cancellation.contains(&"cleanup:drop-decoder-state".to_owned()));
    assert!(events_after_cancellation.contains(&"cleanup:drop-vae-arrays".to_owned()));
    assert!(events_after_cancellation.contains(&"cleanup:synchronize".to_owned()));
    assert_eq!(
        events_after_cancellation.last().map(String::as_str),
        Some("cleanup:clear-cache")
    );
    assert!(
        events_after_cancellation
            .iter()
            .any(|event| event == "decode:complete:middle-attention")
    );
    assert!(
        !events_after_cancellation
            .iter()
            .any(|event| event == "decode:complete:middle-after-attention")
    );
    assert!(
        !events_after_cancellation
            .iter()
            .any(|event| event == "encode")
    );

    engine
        .start_generation(valid_command(64))
        .expect("the cleaned engine should accept a second request");
    let second_png = finish_started_request(&mut engine, 64);
    assert_eq!(second_png, b"parity pixels");
    assert_eq!(
        decode_events(&lifecycle_events)
            .iter()
            .filter(|event| event.as_str() == "decode:complete:input")
            .count(),
        2
    );
}

fn staged_engine(
    lifecycle_events: Arc<Mutex<Vec<String>>>,
    decoded_pixels: Arc<Mutex<Vec<f32>>>,
    decode_stages: &[&str],
) -> Flux2KleinImageEngine {
    Flux2KleinImageEngine::with_components_for_tests(
        FLUX2_KLEIN_OFFICIAL_MODEL_ID,
        Box::new(StagedVaeComponents {
            lifecycle_events,
            decoded_pixels,
            decode_stages: decode_stages
                .iter()
                .map(|stage| (*stage).to_owned())
                .collect(),
            next_decode_stage_index: 0,
        }),
    )
}

fn run_to_completion(mut engine: Flux2KleinImageEngine, request_id: u64) -> Vec<u8> {
    engine.load().expect("the staged component should load");
    engine
        .start_generation(valid_command(request_id))
        .expect("the request should start");
    finish_started_request(&mut engine, request_id)
}

fn finish_started_request(engine: &mut Flux2KleinImageEngine, request_id: u64) -> Vec<u8> {
    for _advance_index in 0..32 {
        let step = engine
            .advance_generation(RequestId::new(request_id))
            .expect("each staged request advancement should succeed");
        if let ImageGenerationEngineStep::Completed {
            generated_image, ..
        } = step
        {
            return generated_image.encoded_bytes;
        }
    }
    panic!("the staged request should complete within its bounded stages");
}

fn advance_to_decoding(engine: &mut Flux2KleinImageEngine, request_id: u64) {
    for _advance_index in 0..6 {
        engine
            .advance_generation(RequestId::new(request_id))
            .expect("conditioning, noise, and four denoising steps should complete");
    }
}

fn valid_command(request_id: u64) -> ImageGenerationCommand {
    ImageGenerationCommand {
        request_id: RequestId::new(request_id),
        model: FLUX2_KLEIN_OFFICIAL_MODEL_ID.to_owned(),
        prompt: "Romeo and Juliet".to_owned(),
        settings: ImageGenerationSettings {
            width_pixels: 64,
            height_pixels: 64,
            steps: 4,
            guidance_thousandths: 1_000,
            seed: request_id,
        },
    }
}

fn cloned_events(lifecycle_events: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    lifecycle_events
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

fn decode_events(lifecycle_events: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    cloned_events(lifecycle_events)
        .into_iter()
        .filter(|event| event.starts_with("decode:"))
        .collect()
}

fn cloned_pixels(decoded_pixels: &Arc<Mutex<Vec<f32>>>) -> Vec<f32> {
    decoded_pixels
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

struct StagedVaeComponents {
    lifecycle_events: Arc<Mutex<Vec<String>>>,
    decoded_pixels: Arc<Mutex<Vec<f32>>>,
    decode_stages: Vec<String>,
    next_decode_stage_index: usize,
}

impl Flux2KleinEngineComponents for StagedVaeComponents {
    fn load(&mut self) -> Result<Flux2KleinComponentLoad, String> {
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
        _seed: u64,
    ) -> Result<Flux2KleinFlowSchedule, String> {
        self.next_decode_stage_index = 0;
        Flux2KleinFlowScheduler::schedule(dimensions.width_pixels(), dimensions.height_pixels())
            .map_err(|error| error.to_string())
    }

    fn condition_prompt(&mut self, _prompt: &str) -> Result<bool, String> {
        Ok(true)
    }

    fn initialize_keyed_noise(&mut self, _seed: u64, _initial_sigma: f64) -> Result<(), String> {
        Ok(())
    }

    fn denoise_euler(
        &mut self,
        _step_index: usize,
        _flow_step: Flux2KleinFlowStep,
    ) -> Result<bool, String> {
        Ok(true)
    }

    fn decode_latents(&mut self) -> Result<bool, String> {
        let stage = self
            .decode_stages
            .get(self.next_decode_stage_index)
            .ok_or_else(|| "decode advanced after pixels were ready".to_owned())?;
        self.record(&format!("decode:{stage}"));
        self.next_decode_stage_index += 1;
        let pixels_are_ready = self.next_decode_stage_index == self.decode_stages.len();
        if pixels_are_ready {
            *self
                .decoded_pixels
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                vec![-0.75, -0.125, 0.25, 0.875];
        }
        Ok(pixels_are_ready)
    }

    fn encode_png(&mut self) -> Result<Vec<u8>, String> {
        self.record("encode");
        Ok(b"parity pixels".to_vec())
    }

    fn finalize_request(
        &mut self,
        outcome: PerformanceAttributionOutcome,
        _encoded_bytes: Option<u64>,
        _failure_description: Option<&str>,
    ) -> Result<(), String> {
        if outcome == PerformanceAttributionOutcome::Cancelled {
            self.record("cleanup:drop-decoder-state");
            self.record("cleanup:drop-tile-graph");
            self.record("cleanup:drop-vae-arrays");
            self.record("cleanup:synchronize");
            self.record("cleanup:clear-cache");
        }
        Ok(())
    }
}

impl StagedVaeComponents {
    fn record(&self, event: &str) {
        self.lifecycle_events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event.to_owned());
    }
}
