//! Scripted Qwen-Image-2.1 component owner shared by lifecycle user-journey tests.

use std::sync::{Arc, Mutex};

use astronomical_ipc_protocol::{
    ImageGenerationCapabilities, ImageGenerationCommand, ImageGenerationSettings, RequestId,
};
use astronomical_model_serving::{
    MlxActiveMemoryBreakdown, MlxMemoryTelemetry, PerformanceAttributionOutcome,
    QWEN_IMAGE_21_OFFICIAL_MODEL_ID, QwenImage21ComponentLoad, QwenImage21EngineComponents,
    QwenImage21EngineRequest, QwenImage21ImageEngine, QwenImage21RenderAdvance,
    QwenImage21Rendered, qwen_image_21_image_generation_capabilities,
};

/// The revision the scripted artifact claims; distinct from any real Hub SHA on purpose so a
/// leaked real identity could never satisfy the fake.
pub(super) const SCRIPTED_REVISION: &str = "0000000000000000000000000000000000000042";

pub(super) fn fake_engine(
    lifecycle_events: Arc<Mutex<Vec<String>>>,
    failing_boundary: Option<&str>,
) -> QwenImage21ImageEngine {
    QwenImage21ImageEngine::with_components_for_tests(
        QWEN_IMAGE_21_OFFICIAL_MODEL_ID,
        Box::new(FakeComponents {
            lifecycle_events,
            failing_boundary: failing_boundary.map(str::to_owned),
            advances: Vec::new(),
            next_advance_index: 0,
            post_cleanup_memory_telemetry: None,
        }),
    )
}

/// A fake whose render boundaries are injected, mirroring how a scripted session reports.
pub(super) fn scripted_advances_engine(
    lifecycle_events: Arc<Mutex<Vec<String>>>,
    advances: Vec<Result<QwenImage21RenderAdvance, String>>,
) -> QwenImage21ImageEngine {
    QwenImage21ImageEngine::with_components_for_tests(
        QWEN_IMAGE_21_OFFICIAL_MODEL_ID,
        Box::new(FakeComponents {
            lifecycle_events,
            failing_boundary: None,
            advances,
            next_advance_index: 0,
            post_cleanup_memory_telemetry: None,
        }),
    )
}

pub(super) fn valid_command(request_id: u64, seed: u64) -> ImageGenerationCommand {
    ImageGenerationCommand {
        request_id: RequestId::new(request_id),
        model: QWEN_IMAGE_21_OFFICIAL_MODEL_ID.to_owned(),
        prompt: "Romeo and Juliet".to_owned(),
        settings: ImageGenerationSettings {
            width_pixels: 1_024,
            height_pixels: 1_024,
            steps: 6,
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
        memory_telemetry.expect("finalized Qwen requests should expose MLX memory");
    assert_eq!(memory_telemetry.allocator_cache_memory_bytes, 0);
    assert!(memory_telemetry.active_memory_bytes <= memory_telemetry.peak_memory_bytes);
    assert_eq!(
        memory_telemetry.active_memory_breakdown,
        MlxActiveMemoryBreakdown::default()
    );
}

/// The boundary sequence one six-step fake render reports.
pub(super) fn six_step_render_advances() -> Vec<Result<QwenImage21RenderAdvance, String>> {
    let mut advances = vec![
        Ok(QwenImage21RenderAdvance::Preparing),
        Ok(QwenImage21RenderAdvance::ConditioningCompleted),
        Ok(QwenImage21RenderAdvance::NoisePrepared),
    ];
    for completed_steps in 1..=6 {
        advances.push(Ok(QwenImage21RenderAdvance::DenoisingStep {
            completed_steps,
            total_steps: 6,
        }));
    }
    advances.push(Ok(QwenImage21RenderAdvance::DecodingCompleted));
    advances.push(Ok(QwenImage21RenderAdvance::Rendered(scripted_rendered())));
    advances
}

pub(super) fn scripted_rendered() -> QwenImage21Rendered {
    QwenImage21Rendered {
        width: 8,
        height: 8,
        // A tiny valid PNG: the engine must encode whatever the session produced.
        rgb_bytes: vec![0_u8; 8 * 8 * 3],
    }
}

struct FakeComponents {
    lifecycle_events: Arc<Mutex<Vec<String>>>,
    failing_boundary: Option<String>,
    advances: Vec<Result<QwenImage21RenderAdvance, String>>,
    next_advance_index: usize,
    post_cleanup_memory_telemetry: Option<MlxMemoryTelemetry>,
}

impl QwenImage21EngineComponents for FakeComponents {
    fn load(&mut self) -> Result<QwenImage21ComponentLoad, String> {
        self.record("load")?;
        Ok(QwenImage21ComponentLoad::new(
            QWEN_IMAGE_21_OFFICIAL_MODEL_ID,
            SCRIPTED_REVISION,
            image_capabilities(),
            400_000_000,
        ))
    }

    fn start_request(
        &mut self,
        request_id: RequestId,
        request: QwenImage21EngineRequest,
    ) -> Result<(), String> {
        self.record(&format!(
            "start:{}:{}x{}:{}:{}",
            request_id.value(),
            request.width_pixels,
            request.height_pixels,
            request.steps,
            request.seed
        ))?;
        self.next_advance_index = 0;
        self.post_cleanup_memory_telemetry = None;
        Ok(())
    }

    fn advance_render(&mut self) -> Result<QwenImage21RenderAdvance, String> {
        let boundary_name = if self.next_advance_index == 0 {
            "prepare".to_owned()
        } else {
            format!("advance:{}", self.next_advance_index)
        };
        self.record(&boundary_name)?;
        let advance = self
            .advances
            .get(self.next_advance_index)
            .cloned()
            .unwrap_or_else(|| Ok(QwenImage21RenderAdvance::Rendered(scripted_rendered())));
        self.next_advance_index += 1;
        advance
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
        if self.failing_boundary.as_deref() == Some(event) {
            Err(format!("injected failure at {event}"))
        } else {
            Ok(())
        }
    }
}

fn image_capabilities() -> ImageGenerationCapabilities {
    qwen_image_21_image_generation_capabilities()
}
