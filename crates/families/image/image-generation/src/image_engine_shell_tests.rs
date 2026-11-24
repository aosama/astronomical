//! The shared image lifecycle through a scripted family: one request at a
//! time, refusals worded with the family's name, a mismatched identifier
//! that leaves the request in progress alone, and an image published only
//! after its finalize succeeds.

use std::time::Instant;

use ipc_protocol::{
    ImageGenerationCapabilities, ImageGenerationCommand, ImageGenerationFailureReason,
    ImageGenerationPhase, ImageGenerationResultMetadata, ImageGenerationSettings, RequestId,
};
use model_primitives::performance_attribution::PerformanceAttributionOutcome;

use super::{
    ActiveImageRequest, ImageComponentLoad, ImageEngineComponents, ImageEngineShell, ImageFamily,
};
use crate::image_generation_engine::{ImageGenerationEngine, ImageGenerationEngineStep};

const GLOBE_MODEL_ID: &str = "Globe-1";
/// The one picture size the Globe family draws.
const GLOBE_SIDE_PIXELS: u32 = 128;

/// Components whose load identity and finalize outcome a test chooses.
struct ScriptedComponents {
    loaded_model_id: String,
    load_failure: Option<String>,
    finalize_succeeds: bool,
    finalized_outcomes: Vec<PerformanceAttributionOutcome>,
}

impl ImageEngineComponents for ScriptedComponents {
    fn load(&mut self) -> Result<ImageComponentLoad, String> {
        if let Some(load_failure) = &self.load_failure {
            return Err(load_failure.clone());
        }
        Ok(ImageComponentLoad::new(
            self.loaded_model_id.clone(),
            "revision-1",
            globe_capabilities(),
            1 << 30,
        ))
    }

    fn finalize_request(
        &mut self,
        outcome: PerformanceAttributionOutcome,
        _published_byte_count: Option<u64>,
        _failure_detail: Option<&str>,
    ) -> Result<(), String> {
        self.finalized_outcomes.push(outcome);
        if self.finalize_succeeds {
            Ok(())
        } else {
            Err("the scripted finalize failed".to_owned())
        }
    }
}

struct GlobeFamily;

/// A two-step render: the first advance reports progress, the second
/// finishes with a three-byte image.
struct GlobeRequest {
    request_id: RequestId,
    completed_steps: u16,
    started_at: Instant,
}

impl ActiveImageRequest for GlobeRequest {
    fn request_id(&self) -> RequestId {
        self.request_id
    }

    fn started_at(&self) -> Instant {
        self.started_at
    }

    fn total_steps(&self) -> u16 {
        2
    }

    fn result_metadata(&self, elapsed_millis: u64) -> ImageGenerationResultMetadata {
        ImageGenerationResultMetadata {
            width_pixels: GLOBE_SIDE_PIXELS,
            height_pixels: GLOBE_SIDE_PIXELS,
            steps: 2,
            guidance_thousandths: 1_000,
            seed: 7,
            elapsed_millis,
        }
    }
}

impl ImageFamily for GlobeFamily {
    type Components = ScriptedComponents;
    type ActiveRequest = GlobeRequest;
    const FAMILY_NAME: &'static str = "Globe";

    fn start_request(
        _engine: &mut ImageEngineShell<Self>,
        generation_command: ImageGenerationCommand,
    ) -> Result<GlobeRequest, ImageGenerationFailureReason> {
        Ok(GlobeRequest {
            request_id: generation_command.request_id,
            completed_steps: 0,
            started_at: Instant::now(),
        })
    }

    fn advance_request(
        engine: &mut ImageEngineShell<Self>,
        mut active_request: GlobeRequest,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason> {
        if active_request.completed_steps == 0 {
            active_request.completed_steps = 1;
            return Ok(engine.restore_with_progress(
                active_request,
                ImageGenerationPhase::Denoising,
                1,
            ));
        }
        engine.complete_rendered_request(active_request, vec![1, 2, 3])
    }
}

fn globe_capabilities() -> ImageGenerationCapabilities {
    ImageGenerationCapabilities {
        minimum_width_pixels: GLOBE_SIDE_PIXELS,
        maximum_width_pixels: GLOBE_SIDE_PIXELS,
        minimum_height_pixels: GLOBE_SIDE_PIXELS,
        maximum_height_pixels: GLOBE_SIDE_PIXELS,
        dimension_multiple_pixels: 16,
        maximum_steps: 2,
        maximum_guidance_thousandths: 1_000,
        output_mime_types: vec!["image/png".to_owned()],
    }
}

fn globe_engine(loaded_model_id: &str, finalize_succeeds: bool) -> ImageEngineShell<GlobeFamily> {
    ImageEngineShell::from_components(
        GLOBE_MODEL_ID,
        Box::new(ScriptedComponents {
            loaded_model_id: loaded_model_id.to_owned(),
            load_failure: None,
            finalize_succeeds,
            finalized_outcomes: Vec::new(),
        }),
    )
}

fn loaded_globe_engine(finalize_succeeds: bool) -> ImageEngineShell<GlobeFamily> {
    let mut engine = globe_engine(GLOBE_MODEL_ID, finalize_succeeds);
    engine.load().expect("the scripted Globe should load");
    engine
}

fn globe_command(request_id: u64) -> ImageGenerationCommand {
    ImageGenerationCommand {
        request_id: RequestId::new(request_id),
        model: GLOBE_MODEL_ID.to_owned(),
        prompt: "All the world's a stage".to_owned(),
        settings: ImageGenerationSettings {
            width_pixels: GLOBE_SIDE_PIXELS,
            height_pixels: GLOBE_SIDE_PIXELS,
            steps: 2,
            guidance_thousandths: 1_000,
            seed: 7,
        },
    }
}

fn fatal(reason: &str) -> ImageGenerationFailureReason {
    ImageGenerationFailureReason::FatalExecution {
        reason: reason.to_owned(),
    }
}

#[test]
fn should_render_one_request_and_publish_it_after_finalizing() {
    let mut engine = loaded_globe_engine(true);
    assert_eq!(engine.loaded_revision(), Some("revision-1"));
    engine
        .start_generation(globe_command(1))
        .expect("the request should start");
    let first_step = engine.advance_generation(RequestId::new(1));
    assert!(matches!(
        first_step,
        Ok(ImageGenerationEngineStep::Progress {
            completed_steps: 1,
            total_steps: 2,
            ..
        })
    ));
    let Ok(ImageGenerationEngineStep::Completed {
        generated_image, ..
    }) = engine.advance_generation(RequestId::new(1))
    else {
        panic!("the second advance should finish the image");
    };
    assert_eq!(generated_image.encoded_bytes, [1, 2, 3]);
    assert_eq!(
        engine.components.finalized_outcomes,
        [PerformanceAttributionOutcome::Success]
    );
}

#[test]
fn should_refuse_to_start_before_load_and_while_a_request_is_active() {
    let mut unloaded_engine = globe_engine(GLOBE_MODEL_ID, true);
    assert_eq!(
        unloaded_engine.start_generation(globe_command(1)),
        Err(fatal("Globe must be loaded before generation starts"))
    );
    let mut engine = loaded_globe_engine(true);
    engine
        .start_generation(globe_command(1))
        .expect("the first request should start");
    assert_eq!(
        engine.start_generation(globe_command(2)),
        Err(ImageGenerationFailureReason::EngineBusy)
    );
    assert_eq!(
        engine.load().err(),
        Some(ImageGenerationFailureReason::EngineBusy)
    );
    assert_eq!(
        engine.update_mlx_memory_limit(1 << 30).err(),
        Some(ImageGenerationFailureReason::EngineBusy)
    );
}

#[test]
fn should_keep_the_active_request_when_another_identifier_advances_or_cancels() {
    let mut engine = loaded_globe_engine(true);
    assert_eq!(
        engine.advance_generation(RequestId::new(1)).err(),
        Some(ImageGenerationFailureReason::invalid_request(
            "no Globe request is active"
        ))
    );
    engine
        .start_generation(globe_command(1))
        .expect("the request should start");
    let mismatch = ImageGenerationFailureReason::invalid_request(
        "request identifier does not match the active Globe request",
    );
    assert_eq!(
        engine.advance_generation(RequestId::new(2)).err(),
        Some(mismatch.clone())
    );
    assert_eq!(engine.cancel_generation(RequestId::new(2)), Err(mismatch));
    assert!(engine.advance_generation(RequestId::new(1)).is_ok());
}

#[test]
fn should_refuse_a_load_under_another_identity_and_bound_a_failure_reason() {
    let mut renamed_engine = globe_engine("Globe-2", true);
    assert_eq!(
        renamed_engine.load().err(),
        Some(fatal("Globe loaded under an unexpected serving identity"))
    );
    let mut failing_engine = globe_engine(GLOBE_MODEL_ID, true);
    failing_engine.components.load_failure = Some("x".repeat(1_000));
    let Some(ImageGenerationFailureReason::FatalExecution { reason }) = failing_engine.load().err()
    else {
        panic!("a failed load should be fatal");
    };
    assert!(reason.starts_with("Globe could not load the selected model: x"));
    assert_eq!(reason.chars().count(), 384);
}

#[test]
fn should_withhold_the_image_when_its_finalize_fails() {
    let mut engine = loaded_globe_engine(false);
    engine
        .start_generation(globe_command(1))
        .expect("the request should start");
    engine
        .advance_generation(RequestId::new(1))
        .expect("the first step should report progress");
    assert_eq!(
        engine.advance_generation(RequestId::new(1)).err(),
        Some(fatal("Globe could not finalize the generated image"))
    );
}

#[test]
fn should_cancel_quietly_when_idle_and_report_a_failed_cleanup() {
    let mut engine = loaded_globe_engine(false);
    assert_eq!(engine.cancel_generation(RequestId::new(1)), Ok(()));
    engine
        .start_generation(globe_command(1))
        .expect("the request should start");
    assert_eq!(
        engine.cancel_generation(RequestId::new(1)),
        Err(fatal("Globe could not cleanly cancel image generation"))
    );
    assert_eq!(
        engine.components.finalized_outcomes,
        [PerformanceAttributionOutcome::Cancelled]
    );
    assert_eq!(
        engine.fail_active_request("pressure".to_owned(), "Globe stalled"),
        fatal("Globe image generation and cleanup failed")
    );
}
