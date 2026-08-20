#![allow(dead_code)]

use std::{
    future::Future,
    io::Cursor,
    pin::Pin,
    sync::{Arc, Mutex},
};

use astronomical_ipc_protocol::MtpRuntimeState;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatModelCapabilities, GeneratedImage, ImageGenerationCommand,
    ImageGenerationResultMetadata,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationStartError,
    ImageGenerationExecutionError, ImageGenerationOutput, WorkerActivity, WorkerHealthSnapshot,
    WorkerHealthStatus,
};
use image::{
    ExtendedColorType, ImageEncoder,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use tokio::sync::mpsc;

pub(crate) mod supervisor;

pub const MODEL_ID: &str = "astronomical/application-test-model";

pub struct ScriptedExecutor {
    pub health_snapshot: WorkerHealthSnapshot,
    pub stream_events: Vec<ChatGenerationStreamEvent>,
    received_generation_commands: Arc<Mutex<Vec<ChatGenerationCommand>>>,
    received_image_generation_commands: Arc<Mutex<Vec<ImageGenerationCommand>>>,
    pub image_generation_outcome: Result<ImageGenerationOutput, ImageGenerationExecutionError>,
    /// Test override: when true, the executor reports a busy worker so the
    /// config-reload endpoint returns HTTP 409.
    pub is_busy_override: bool,
}

impl ScriptedExecutor {
    pub fn ready(stream_events: Vec<ChatGenerationStreamEvent>) -> Self {
        Self {
            health_snapshot: WorkerHealthSnapshot::ready_with_model(
                MODEL_ID.to_owned(),
                ChatModelCapabilities {
                    supports_reasoning: true,
                    supports_tool_calls: true,
                    has_vision: true,
                    max_input_tokens: 241_664,
                    max_output_tokens: 20_480,
                    context_window: 262_144,
                },
                MtpRuntimeState::Disabled,
                None,
            ),
            stream_events,
            received_generation_commands: Arc::new(Mutex::new(Vec::new())),
            received_image_generation_commands: Arc::new(Mutex::new(Vec::new())),
            image_generation_outcome: Ok(ImageGenerationOutput {
                generated_image: GeneratedImage {
                    mime_type: "image/png".to_owned(),
                    encoded_bytes: valid_png_bytes(1_024, 1_024),
                },
                result_metadata: ImageGenerationResultMetadata {
                    width_pixels: 1_024,
                    height_pixels: 1_024,
                    steps: 4,
                    guidance_thousandths: 1_000,
                    seed: 7,
                    elapsed_millis: 25,
                },
            }),
            is_busy_override: false,
        }
    }

    pub fn received_generation_commands(&self) -> Arc<Mutex<Vec<ChatGenerationCommand>>> {
        Arc::clone(&self.received_generation_commands)
    }

    pub fn received_image_generation_commands(&self) -> Arc<Mutex<Vec<ImageGenerationCommand>>> {
        Arc::clone(&self.received_image_generation_commands)
    }

    pub fn unavailable() -> Self {
        let mut executor = Self::ready(Vec::new());
        executor.health_snapshot =
            WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable);
        executor
    }
}

fn valid_png_bytes(width_pixels: u32, height_pixels: u32) -> Vec<u8> {
    let rgb_bytes = vec![0; width_pixels as usize * height_pixels as usize * 3];
    let mut png_bytes = Vec::new();
    PngEncoder::new_with_quality(
        Cursor::new(&mut png_bytes),
        CompressionType::Best,
        FilterType::Adaptive,
    )
    .write_image(
        &rgb_bytes,
        width_pixels,
        height_pixels,
        ExtendedColorType::Rgb8,
    )
    .expect("the scripted RGB image should encode as PNG");
    png_bytes
}

impl ChatGenerationExecutor for ScriptedExecutor {
    fn start_chat_generation(
        &self,
        generation_command: ChatGenerationCommand,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<ChatGenerationStreamEvent>,
                        GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            self.received_generation_commands
                .lock()
                .expect("the scripted executor command log should not be poisoned")
                .push(generation_command);
            let (stream_event_sender, stream_event_receiver) =
                mpsc::channel(self.stream_events.len().max(1));
            for stream_event in &self.stream_events {
                stream_event_sender
                    .send(stream_event.clone())
                    .await
                    .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            }
            Ok(stream_event_receiver)
        })
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        let mut worker_health_snapshot = self.health_snapshot.clone();
        if self.is_busy_override {
            worker_health_snapshot.activity = WorkerActivity::Generating;
        }
        worker_health_snapshot
    }
}

impl astronomical_supervisor::ImageGenerationExecutor for ScriptedExecutor {
    fn start_image_generation(
        &self,
        image_generation_command: ImageGenerationCommand,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<
                            Result<ImageGenerationOutput, ImageGenerationExecutionError>,
                        >,
                        GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            self.received_image_generation_commands
                .lock()
                .expect("the scripted image command log should not be poisoned")
                .push(image_generation_command);
            let (image_result_sender, image_result_receiver) = mpsc::channel(1);
            image_result_sender
                .send(self.image_generation_outcome.clone())
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            Ok(image_result_receiver)
        })
    }
}
