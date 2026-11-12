#![allow(dead_code)]

use std::{
    future::Future,
    io::Cursor,
    pin::Pin,
    sync::{Arc, Mutex},
};

use astronomical_daemon::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, DecisionsExecutionError, DecisionsOutput,
    EmbeddingsExecutionError, EmbeddingsOutput, GenerationStartError,
    ImageGenerationExecutionError, ImageGenerationOutput, SpeechExecutionError, SpeechOutput,
    WorkerActivity, WorkerHealthSnapshot, WorkerHealthStatus,
};
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatModelCapabilities, DecisionsCommand, EmbeddingsCommand,
    GeneratedImage, ImageGenerationCapabilities, ImageGenerationCommand,
    ImageGenerationResultMetadata, SpeechCommand, SpeechResultMetadata, WorkerModelCapabilities,
};
use image::{
    ExtendedColorType, ImageEncoder,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use tokio::sync::mpsc;

pub(crate) mod daemon;

pub const MODEL_ID: &str = "astronomical/application-test-model";

pub struct ScriptedExecutor {
    pub health_snapshot: WorkerHealthSnapshot,
    pub stream_events: Vec<ChatGenerationStreamEvent>,
    received_generation_commands: Arc<Mutex<Vec<ChatGenerationCommand>>>,
    received_image_generation_commands: Arc<Mutex<Vec<ImageGenerationCommand>>>,
    received_embeddings_commands: Arc<Mutex<Vec<EmbeddingsCommand>>>,
    pub image_generation_outcome: Result<ImageGenerationOutput, ImageGenerationExecutionError>,
    pub embeddings_outcome: Result<EmbeddingsOutput, EmbeddingsExecutionError>,
    received_decisions_commands: Arc<Mutex<Vec<DecisionsCommand>>>,
    pub decisions_outcome: Result<DecisionsOutput, DecisionsExecutionError>,
    received_speech_commands: Arc<Mutex<Vec<SpeechCommand>>>,
    pub speech_outcome: Result<SpeechOutput, SpeechExecutionError>,
    /// Test override: when true, the executor reports a busy worker so the
    /// config-reload endpoint returns HTTP 409.
    pub is_busy_override: bool,
}

impl ScriptedExecutor {
    pub fn ready(stream_events: Vec<ChatGenerationStreamEvent>) -> Self {
        Self {
            health_snapshot: WorkerHealthSnapshot::ready_with_model(
                MODEL_ID.to_owned(),
                WorkerModelCapabilities::chat_and_image(
                    ChatModelCapabilities {
                        supports_reasoning: true,
                        supports_tool_calls: true,
                        has_vision: true,
                        max_input_tokens: 241_664,
                        max_output_tokens: 20_480,
                        context_window: 262_144,
                    },
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
                ),
            ),
            stream_events,
            received_generation_commands: Arc::new(Mutex::new(Vec::new())),
            received_image_generation_commands: Arc::new(Mutex::new(Vec::new())),
            received_embeddings_commands: Arc::new(Mutex::new(Vec::new())),
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
            received_decisions_commands: Arc::new(Mutex::new(Vec::new())),
            decisions_outcome: Ok(DecisionsOutput {
                question_outcomes: Vec::new(),
                input_token_count: 0,
                elapsed_millis: 4,
            }),
            embeddings_outcome: Ok(EmbeddingsOutput {
                embeddings: vec![unit_embedding_fixture()],
                input_token_counts: vec![8],
                elapsed_millis: 12,
            }),
            received_speech_commands: Arc::new(Mutex::new(Vec::new())),
            speech_outcome: Ok(SpeechOutput {
                pcm16_bytes: vec![0x00, 0x00, 0x00, 0x40],
                result_metadata: SpeechResultMetadata {
                    sample_rate_hertz: 24_000,
                    sample_count: 2,
                    segment_count: 1,
                    cut_short_segment_count: 0,
                    voice_name: "af_heart".to_owned(),
                    seed: 42,
                },
            }),
        }
    }

    pub fn received_speech_commands(&self) -> Arc<Mutex<Vec<SpeechCommand>>> {
        Arc::clone(&self.received_speech_commands)
    }

    pub fn received_decisions_commands(&self) -> Arc<Mutex<Vec<DecisionsCommand>>> {
        Arc::clone(&self.received_decisions_commands)
    }

    pub fn received_embeddings_commands(&self) -> Arc<Mutex<Vec<EmbeddingsCommand>>> {
        Arc::clone(&self.received_embeddings_commands)
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

fn unit_embedding_fixture() -> Vec<f32> {
    let mut components = vec![0.0_f32; 768];
    components[0] = 1.0;
    components
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

impl astronomical_daemon::ImageGenerationExecutor for ScriptedExecutor {
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

    fn start_embeddings_generation(
        &self,
        embeddings_command: EmbeddingsCommand,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<Result<EmbeddingsOutput, EmbeddingsExecutionError>>,
                        GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            self.received_embeddings_commands
                .lock()
                .expect("the scripted embeddings command log should not be poisoned")
                .push(embeddings_command);
            let (embeddings_result_sender, embeddings_result_receiver) = mpsc::channel(1);
            embeddings_result_sender
                .send(self.embeddings_outcome.clone())
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            Ok(embeddings_result_receiver)
        })
    }

    fn start_decisions_generation(
        &self,
        decisions_command: DecisionsCommand,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<Result<DecisionsOutput, DecisionsExecutionError>>,
                        GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            self.received_decisions_commands
                .lock()
                .expect("the scripted decisions command log should not be poisoned")
                .push(decisions_command);
            let (decisions_result_sender, decisions_result_receiver) = mpsc::channel(1);
            decisions_result_sender
                .send(self.decisions_outcome.clone())
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            Ok(decisions_result_receiver)
        })
    }

    fn start_speech_generation(
        &self,
        speech_command: SpeechCommand,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<Result<SpeechOutput, SpeechExecutionError>>,
                        GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            self.received_speech_commands
                .lock()
                .expect("the scripted speech command log should not be poisoned")
                .push(speech_command);
            let (speech_result_sender, speech_result_receiver) = mpsc::channel(1);
            speech_result_sender
                .send(self.speech_outcome.clone())
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            Ok(speech_result_receiver)
        })
    }
}

pub(crate) mod daemon_ipc;
