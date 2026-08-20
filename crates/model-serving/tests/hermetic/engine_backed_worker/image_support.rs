//! Scripted image runtime and protocol harness shared by worker image journeys.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use astronomical_ipc_protocol::{
    GeneratedImage, ImageGenerationCapabilities, ImageGenerationCommand,
    ImageGenerationFailureReason, ImageGenerationPhase, ImageGenerationResultMetadata,
    ImageGenerationSettings, MAX_IPC_FRAME_BYTES, MlxMemorySnapshotSource, ProtocolReader,
    ProtocolWriter, RequestId, WorkerCommand, WorkerEvent, WorkerFlux2KleinModelConfiguration,
    WorkerImageGenerationModelFamily, WorkerModelConfiguration,
};
use astronomical_model_serving::{
    EngineBackedWorker, FLUX2_KLEIN_OFFICIAL_MODEL_ID, FLUX2_KLEIN_OFFICIAL_REVISION,
    ImageGenerationEngine, ImageGenerationEngineLoadResult, ImageGenerationEngineStep,
    MlxActiveMemoryBreakdown, MlxMemoryLimitAdjustment, MlxMemoryTelemetry, ModelFactory,
    ModelFactoryRuntime, WorkerRuntimeError,
};
use image::ImageEncoder;
use tokio::io::{AsyncWrite, duplex, split};
use tokio::task::JoinHandle;
use tokio::time::timeout;

use super::chat::scripted_chat_test_doubles::{ScriptedChatEngine, ScriptedChatProcessor};

pub(super) type ScriptedRuntime =
    ModelFactoryRuntime<ScriptedChatProcessor, ScriptedChatEngine, ScriptedImageEngine>;

pub(super) struct ScriptedRuntimeFactory {
    runtimes: Mutex<VecDeque<Result<ScriptedRuntime, String>>>,
}

impl ScriptedRuntimeFactory {
    pub(super) fn new(runtimes: Vec<Result<ScriptedRuntime, String>>) -> Self {
        Self {
            runtimes: Mutex::new(runtimes.into()),
        }
    }
}

impl ModelFactory<ScriptedChatProcessor, ScriptedChatEngine, ScriptedImageEngine>
    for ScriptedRuntimeFactory
{
    async fn create(
        &self,
        _model_directory: &str,
        _model_configuration: WorkerModelConfiguration,
    ) -> Result<ScriptedRuntime, String> {
        lock(&self.runtimes)
            .pop_front()
            .expect("the test should configure every model creation")
    }
}

pub(super) struct ScriptedImageEngine {
    request_scripts: VecDeque<Vec<Result<ImageGenerationEngineStep, ImageGenerationFailureReason>>>,
    active_steps: VecDeque<Result<ImageGenerationEngineStep, ImageGenerationFailureReason>>,
    pub(super) cancellation_count: Arc<Mutex<usize>>,
    pub(super) updated_limits: Arc<Mutex<Vec<u64>>>,
    post_cleanup_memory_telemetry: Option<MlxMemoryTelemetry>,
    load_failure_reason: Option<String>,
    cancellation_failure_reason: Option<String>,
}

impl ScriptedImageEngine {
    pub(super) fn new(
        request_scripts: Vec<Vec<Result<ImageGenerationEngineStep, ImageGenerationFailureReason>>>,
    ) -> Self {
        Self {
            request_scripts: request_scripts.into(),
            active_steps: VecDeque::new(),
            cancellation_count: Arc::new(Mutex::new(0)),
            updated_limits: Arc::new(Mutex::new(Vec::new())),
            post_cleanup_memory_telemetry: None,
            load_failure_reason: None,
            cancellation_failure_reason: None,
        }
    }

    pub(super) fn failing_load(load_failure_reason: String) -> Self {
        let mut engine = Self::new(Vec::new());
        engine.load_failure_reason = Some(load_failure_reason);
        engine
    }

    pub(super) fn failing_cancellation(
        request_scripts: Vec<Vec<Result<ImageGenerationEngineStep, ImageGenerationFailureReason>>>,
        cancellation_failure_reason: String,
    ) -> Self {
        let mut engine = Self::new(request_scripts);
        engine.cancellation_failure_reason = Some(cancellation_failure_reason);
        engine
    }

    fn finalize(&mut self) {
        self.active_steps.clear();
        self.post_cleanup_memory_telemetry = Some(finalized_memory_telemetry());
    }
}

impl ImageGenerationEngine for ScriptedImageEngine {
    fn load(&mut self) -> Result<ImageGenerationEngineLoadResult, ImageGenerationFailureReason> {
        if let Some(load_failure_reason) = self.load_failure_reason.take() {
            return Err(ImageGenerationFailureReason::FatalExecution {
                reason: load_failure_reason,
            });
        }
        Ok(ImageGenerationEngineLoadResult::new(
            FLUX2_KLEIN_OFFICIAL_MODEL_ID,
            image_capabilities(),
        )
        .with_minimum_mlx_memory_ceiling_bytes(1_000))
    }

    fn start_generation(
        &mut self,
        _generation_command: ImageGenerationCommand,
    ) -> Result<(), ImageGenerationFailureReason> {
        self.active_steps = self
            .request_scripts
            .pop_front()
            .ok_or_else(|| ImageGenerationFailureReason::FatalExecution {
                reason: "no scripted image request remains".to_owned(),
            })?
            .into();
        self.post_cleanup_memory_telemetry = None;
        Ok(())
    }

    fn advance_generation(
        &mut self,
        _request_id: RequestId,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason> {
        let step = self
            .active_steps
            .pop_front()
            .expect("a scripted step should remain");
        if matches!(step, Ok(ImageGenerationEngineStep::Completed { .. })) || step.is_err() {
            self.finalize();
        }
        step
    }

    fn cancel_generation(
        &mut self,
        _request_id: RequestId,
    ) -> Result<(), ImageGenerationFailureReason> {
        *lock(&self.cancellation_count) += 1;
        if let Some(cancellation_failure_reason) = self.cancellation_failure_reason.take() {
            return Err(ImageGenerationFailureReason::FatalExecution {
                reason: cancellation_failure_reason,
            });
        }
        self.finalize();
        Ok(())
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.post_cleanup_memory_telemetry.take()
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        Some(finalized_memory_telemetry())
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, ImageGenerationFailureReason> {
        lock(&self.updated_limits).push(requested_mlx_memory_ceiling_bytes);
        Ok(MlxMemoryLimitAdjustment::new(
            requested_mlx_memory_ceiling_bytes,
            requested_mlx_memory_ceiling_bytes,
            1_000,
            astronomical_ipc_protocol::ExpertMemoryMode::Resident,
            Some(finalized_memory_telemetry()),
        ))
    }
}

pub(super) fn image_configuration() -> WorkerModelConfiguration {
    WorkerModelConfiguration::Flux2Klein(WorkerFlux2KleinModelConfiguration {
        model_id: FLUX2_KLEIN_OFFICIAL_MODEL_ID.to_owned(),
        model_family: WorkerImageGenerationModelFamily::Flux2Klein,
        artifact_revision: FLUX2_KLEIN_OFFICIAL_REVISION.to_owned(),
    })
}

pub(super) fn swap_image_command() -> WorkerCommand {
    WorkerCommand::SwapModel {
        model_directory: "/models/image".to_owned(),
        model_configuration: image_configuration(),
    }
}

pub(super) fn image_command(request_number: u64) -> ImageGenerationCommand {
    ImageGenerationCommand {
        request_id: RequestId::new(request_number),
        model: FLUX2_KLEIN_OFFICIAL_MODEL_ID.to_owned(),
        prompt: "Romeo and Juliet beneath a night sky".to_owned(),
        settings: ImageGenerationSettings {
            width_pixels: 512,
            height_pixels: 512,
            steps: 2,
            guidance_thousandths: 3_500,
            seed: 17,
        },
    }
}

fn image_capabilities() -> ImageGenerationCapabilities {
    ImageGenerationCapabilities {
        minimum_width_pixels: 256,
        maximum_width_pixels: 2_048,
        minimum_height_pixels: 256,
        maximum_height_pixels: 2_048,
        dimension_multiple_pixels: 16,
        maximum_steps: 50,
        maximum_guidance_thousandths: 20_000,
        output_mime_types: vec!["image/png".to_owned()],
    }
}

pub(super) fn progress_step() -> ImageGenerationEngineStep {
    ImageGenerationEngineStep::Progress {
        phase: ImageGenerationPhase::Denoising,
        completed_steps: 1,
        total_steps: 2,
        elapsed_millis: 4,
    }
}

pub(super) fn completed_step() -> ImageGenerationEngineStep {
    ImageGenerationEngineStep::Completed {
        generated_image: GeneratedImage {
            mime_type: "image/png".to_owned(),
            encoded_bytes: valid_png_bytes(512, 512),
        },
        result_metadata: ImageGenerationResultMetadata {
            width_pixels: 512,
            height_pixels: 512,
            steps: 2,
            guidance_thousandths: 3_500,
            seed: 17,
            elapsed_millis: 9,
        },
    }
}

fn finalized_memory_telemetry() -> MlxMemoryTelemetry {
    MlxMemoryTelemetry::new(96, 0, 512, MlxActiveMemoryBreakdown::default())
}

pub(super) fn assert_finalized_with_cleanup(event: WorkerEvent, request_number: u64) {
    assert!(
        matches!(event, WorkerEvent::ImageGenerationFinalized { request_id, mlx_memory_snapshot: Some(snapshot), .. } if request_id == RequestId::new(request_number) && snapshot.source == MlxMemorySnapshotSource::Finalized && snapshot.allocator_cache_memory_bytes == 0)
    );
}

pub(super) fn assert_completed_payload(event: WorkerEvent, request_number: u64) {
    assert!(
        matches!(event, WorkerEvent::ImageGenerationCompleted { request_id, generated_image, .. } if request_id == RequestId::new(request_number) && generated_image.encoded_bytes == valid_png_bytes(512, 512))
    );
}

fn valid_png_bytes(width_pixels: u32, height_pixels: u32) -> Vec<u8> {
    let rgb_bytes = vec![0; width_pixels as usize * height_pixels as usize * 3];
    let mut png_bytes = Vec::new();
    image::codecs::png::PngEncoder::new_with_quality(
        std::io::Cursor::new(&mut png_bytes),
        image::codecs::png::CompressionType::Best,
        image::codecs::png::FilterType::Adaptive,
    )
    .write_image(
        &rgb_bytes,
        width_pixels,
        height_pixels,
        image::ExtendedColorType::Rgb8,
    )
    .expect("the scripted RGB image should encode as PNG");
    png_bytes
}

pub(super) async fn start_idle_worker(
    runtimes: Vec<Result<ScriptedRuntime, String>>,
    memory_ceiling_bytes: u64,
) -> (
    ProtocolReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>,
    ProtocolWriter<tokio::io::WriteHalf<tokio::io::DuplexStream>>,
    JoinHandle<Result<(), WorkerRuntimeError>>,
) {
    start_worker(EngineBackedWorker::idle_with_model_factory(
        ScriptedRuntimeFactory::new(runtimes),
        memory_ceiling_bytes,
    ))
    .await
}

pub(super) async fn start_worker(
    worker: EngineBackedWorker<
        ScriptedChatProcessor,
        ScriptedChatEngine,
        ScriptedRuntimeFactory,
        ScriptedImageEngine,
    >,
) -> (
    ProtocolReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>,
    ProtocolWriter<tokio::io::WriteHalf<tokio::io::DuplexStream>>,
    JoinHandle<Result<(), WorkerRuntimeError>>,
) {
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_read, supervisor_write) = split(supervisor_transport);
    let (worker_read, worker_write) = split(worker_transport);
    let worker_task = tokio::spawn(async move { worker.run(worker_read, worker_write).await });
    (
        ProtocolReader::new(supervisor_read),
        ProtocolWriter::new(supervisor_write),
        worker_task,
    )
}

pub(super) async fn next_event<ReadTransport>(
    reader: &mut ProtocolReader<ReadTransport>,
) -> WorkerEvent
where
    ReadTransport: tokio::io::AsyncRead + Unpin,
{
    reader
        .next_event()
        .await
        .expect("valid event")
        .expect("open worker")
}

pub(super) async fn close_worker<WriteTransport>(
    writer: ProtocolWriter<WriteTransport>,
    worker: JoinHandle<Result<(), WorkerRuntimeError>>,
) where
    WriteTransport: AsyncWrite + Unpin,
{
    writer.close().await.expect("close transport");
    assert!(
        timeout(Duration::from_secs(1), worker)
            .await
            .expect("worker stop")
            .expect("worker join")
            .is_ok()
    );
}

pub(super) fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
