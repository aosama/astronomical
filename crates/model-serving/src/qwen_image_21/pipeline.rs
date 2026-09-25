//! The Qwen-Image-2.1 text-to-image render pipeline.
//!
//! Port of the reference `QwenImage21Pipeline.__call__` for the default configuration:
//! text-only prompt, no classifier-free guidance, no KV cache, no tiling. Every component is
//! the family's proven MLX port: the artifact tokenizer, the Qwen3-VL text encoder, the
//! block-causal denoising transformer, the flow-matching schedule, and the VAE decoder.
//!
//! The render itself lives in `render_session`: a `QwenImage21Pipeline` is consumed into a
//! `QwenImage21RenderSession` that executes the reference's sequence one boundary at a time —
//! conditioning, noise, one Euler step per step, then decode — releasing each component in the
//! reference's offload order (`text_encoder->transformer->vae`) as its phase ends. That release
//! is what keeps a 1024×1024 decode inside the wired-memory ceiling, and it is also why one
//! pipeline serves exactly one render: `render` consumes the pipeline, so a second render
//! reloads rather than silently running without weights.

use std::fs::File;
use std::path::Path;

use astronomical_runtime_integration::MlxRuntime;
use tokenizers::Tokenizer;

use crate::qwen_image_21::engine_error::QwenImage21EngineError;
use crate::qwen_image_21::render_contract::{
    QwenImage21RenderAdvance, QwenImage21RenderRequest, QwenImage21Rendered,
};
use crate::qwen_image_21::{QwenImage21TextEncoder, QwenImage21Transformer, QwenImage21VaeDecoder};

/// The Qwen3-VL text encoder's hidden width, which is also the denoising transformer's
/// `context_in_dim`: the sliced encoder output is handed to the transformer unchanged, so the
/// pipeline needs the width to describe the slice. The family constant carries the coupling —
/// the transformer's `CONTEXT_INPUT_WIDTH` and the encoder's `HIDDEN_WIDTH` read it too.
pub(in crate::qwen_image_21) const TEXT_EMBEDDING_WIDTH: i32 =
    crate::qwen_image_21::QWEN_IMAGE_21_TEXT_EMBEDDING_WIDTH as i32;

/// The loaded text-to-image pipeline: tokenizer, text encoder, transformer, and VAE decoder.
///
/// Constructing it maps every component's weights; rendering consumes it, because the render
/// releases the encoder after conditioning and the transformer after denoising (the reference's
/// offload order) so the VAE's full-resolution decode runs with only the VAE weights resident.
/// Reload the pipeline to render again.
pub struct QwenImage21Pipeline {
    pub(in crate::qwen_image_21) tokenizer: Tokenizer,
    pub(in crate::qwen_image_21) encoder: QwenImage21TextEncoder,
    pub(in crate::qwen_image_21) transformer: QwenImage21Transformer,
    pub(in crate::qwen_image_21) decoder: QwenImage21VaeDecoder,
}

impl QwenImage21Pipeline {
    /// Loads every component from an installed artifact root (the `snapshots/<sha>` directory).
    pub fn load(
        runtime: &MlxRuntime,
        artifact_root: &Path,
    ) -> Result<Self, QwenImage21EngineError> {
        let tokenizer_path = artifact_root.join("processor").join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|source| {
            QwenImage21EngineError::Execution {
                description: format!(
                    "the artifact tokenizer at {} failed to load: {source}",
                    tokenizer_path.display()
                ),
            }
        })?;
        let encoder = QwenImage21TextEncoder::load(
            runtime,
            File::open(artifact_root.join("text_encoder").join("model.safetensors")).map_err(
                |source| QwenImage21EngineError::Execution {
                    description: format!("the text-encoder weights failed to open: {source}"),
                },
            )?,
        )?;
        let transformer = QwenImage21Transformer::load(
            runtime,
            File::open(artifact_root.join("transformer").join("model.safetensors")).map_err(
                |source| QwenImage21EngineError::Execution {
                    description: format!("the transformer weights failed to open: {source}"),
                },
            )?,
        )?;
        let decoder = QwenImage21VaeDecoder::load(
            runtime,
            File::open(artifact_root.join("vae").join("model.safetensors")).map_err(|source| {
                QwenImage21EngineError::Execution {
                    description: format!("the VAE weights failed to open: {source}"),
                }
            })?,
        )
        .map_err(|source| QwenImage21EngineError::Execution {
            description: format!("the VAE decoder failed to load: {source}"),
        })?;
        Ok(Self {
            tokenizer,
            encoder,
            transformer,
            decoder,
        })
    }

    /// Renders one image; `step_observer` receives `(step_index, total_steps)` once per
    /// denoising step so callers can report live progress.
    ///
    /// The synchronous loop over [`Self::into_render_session`]: the serving engine steps the
    /// same session, so both execute identical arithmetic. The Euler update
    /// `latents + (sigma_next - sigma) · model_output` runs on the GPU arrays, one evaluation
    /// per denoising step.
    pub fn render(
        self,
        runtime: &MlxRuntime,
        request: &QwenImage21RenderRequest,
        step_observer: &mut dyn FnMut(usize, usize),
    ) -> Result<QwenImage21Rendered, QwenImage21EngineError> {
        let mut session = self.into_render_session(request)?;
        loop {
            match session.advance(runtime)? {
                QwenImage21RenderAdvance::Rendered(rendered) => return Ok(rendered),
                QwenImage21RenderAdvance::DenoisingStep {
                    completed_steps,
                    total_steps,
                } => step_observer(completed_steps - 1, total_steps),
                _ => {}
            }
        }
    }
}
