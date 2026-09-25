//! One render in progress, driven one bounded boundary at a time.
//!
//! The session is the single render path: `QwenImage21Pipeline::render` loops it synchronously
//! for one-shot callers (the example, the acceptance journeys), while the serving engine pulls
//! `advance` once per boundary so a worker stays responsive between denoising steps. Both
//! therefore execute exactly the same arithmetic in the same order — a journey passing is a
//! proof for both.
//!
//! Boundaries follow the reference's component offload order (`text_encoder->transformer->vae`):
//! conditioning drops the encoder, the decode boundary drops the transformer, and the VAE decode
//! then owns the working set with only the VAE's weights resident. Dropping mid-sequence is what
//! keeps a full-resolution decode inside the wired-memory ceiling.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};
use tokenizers::Tokenizer;

use crate::qwen_image_21::pipeline::{QwenImage21Pipeline, TEXT_EMBEDDING_WIDTH};
use crate::qwen_image_21::render_contract::{
    QwenImage21RenderAdvance, QwenImage21RenderRequest, QwenImage21Rendered, rgba_to_rgb8,
};
use crate::qwen_image_21::{
    FlowMatchSchedule, FlowMatchSchedulerParams, QWEN_IMAGE_21_LATENT_CHANNEL_COUNT,
    QWEN_IMAGE_21_SYS_PROMPT, QwenImage21EngineError, QwenImage21TextEncoder,
    QwenImage21Transformer, QwenImage21TransformerRequest, QwenImage21VaeDecoder,
    build_prompt_conditioning, build_schedule, default_shift, latent_spatial_dimensions,
    normalize_empty_prompt, render_t2i_prompt_template, resolve_generation_dimensions,
    special_tokens,
};

/// Where a session is in the reference's render sequence.
enum RenderPhase {
    /// Tokenize, encode, and slice the prompt; then release the encoder.
    Conditioning,
    /// Resolve geometry, build the schedule, draw the seeded noise.
    Noise,
    /// One flow-matching Euler step per `advance` call.
    Denoising,
    /// Unpack the latents, release the transformer, and decode to clamped RGBA pixels.
    Decoding,
    /// Convert the clamped pixels to RGB bytes.
    Converting,
    /// The render produced its image; the session is spent.
    Completed,
}

/// A render that owns every pipeline component it still needs.
///
/// Constructed by consuming a freshly loaded `QwenImage21Pipeline`; the session releases each
/// component as its phase ends, so a session is single-use by construction exactly like the
/// reference's offload sequence.
pub(in crate::qwen_image_21) struct QwenImage21RenderSession {
    request: OwnedRenderRequest,
    tokenizer: Tokenizer,
    encoder: Option<QwenImage21TextEncoder>,
    transformer: Option<QwenImage21Transformer>,
    decoder: QwenImage21VaeDecoder,
    phase: RenderPhase,
    step_index: usize,
    sliced_hidden: Option<MlxArray>,
    vlm_image_mask: Vec<bool>,
    img_shapes: [(usize, usize); 1],
    latent_height: usize,
    latent_width: usize,
    rendered_width: usize,
    rendered_height: usize,
    schedule: Option<FlowMatchSchedule>,
    latents: Option<MlxArray>,
    pixel_values: Option<Vec<f32>>,
}

/// The request fields the session owns after `QwenImage21RenderRequest` is checked in.
struct OwnedRenderRequest {
    prompt: String,
    width: usize,
    height: usize,
    num_inference_steps: usize,
    seed: u64,
}

impl QwenImage21Pipeline {
    /// Consumes the pipeline into a session that renders `request` one boundary per `advance`.
    ///
    /// `render` is the synchronous loop over this; the serving engine is the stepping consumer.
    /// The pipeline is consumed because the render releases components in the reference's
    /// offload order as it advances, so a pipeline cannot serve a second render anyway.
    pub(in crate::qwen_image_21) fn into_render_session(
        self,
        request: &QwenImage21RenderRequest,
    ) -> Result<QwenImage21RenderSession, QwenImage21EngineError> {
        if request.num_inference_steps == 0 {
            return Err(QwenImage21EngineError::InvalidInput {
                description: "a render needs at least one denoising step".to_owned(),
            });
        }
        let QwenImage21Pipeline {
            tokenizer,
            encoder,
            transformer,
            decoder,
        } = self;
        Ok(QwenImage21RenderSession {
            request: OwnedRenderRequest {
                prompt: request.prompt.clone(),
                width: request.width,
                height: request.height,
                num_inference_steps: request.num_inference_steps,
                seed: request.seed,
            },
            tokenizer,
            encoder: Some(encoder),
            transformer: Some(transformer),
            decoder,
            phase: RenderPhase::Conditioning,
            step_index: 0,
            sliced_hidden: None,
            vlm_image_mask: Vec::new(),
            img_shapes: [(0, 0); 1],
            latent_height: 0,
            latent_width: 0,
            rendered_width: 0,
            rendered_height: 0,
            schedule: None,
            latents: None,
            pixel_values: None,
        })
    }
}

impl QwenImage21RenderSession {
    /// Executes exactly one boundary of the reference's render sequence.
    pub(in crate::qwen_image_21) fn advance(
        &mut self,
        runtime: &MlxRuntime,
    ) -> Result<QwenImage21RenderAdvance, QwenImage21EngineError> {
        match self.phase {
            RenderPhase::Conditioning => {
                let sliced_hidden = self.condition_prompt(runtime)?;
                self.sliced_hidden = Some(sliced_hidden);
                self.phase = RenderPhase::Noise;
                Ok(QwenImage21RenderAdvance::ConditioningCompleted)
            }
            RenderPhase::Noise => {
                self.prepare_noise(runtime)?;
                self.phase = RenderPhase::Denoising;
                Ok(QwenImage21RenderAdvance::NoisePrepared)
            }
            RenderPhase::Denoising => {
                let completed_steps = self.denoise_one_step(runtime)?;
                if self.step_index == self.request.num_inference_steps {
                    self.phase = RenderPhase::Decoding;
                }
                Ok(QwenImage21RenderAdvance::DenoisingStep {
                    completed_steps,
                    total_steps: self.request.num_inference_steps,
                })
            }
            RenderPhase::Decoding => {
                self.decode_latents(runtime)?;
                self.phase = RenderPhase::Converting;
                Ok(QwenImage21RenderAdvance::DecodingCompleted)
            }
            RenderPhase::Converting => {
                let rendered = self.convert_pixels();
                self.phase = RenderPhase::Completed;
                Ok(QwenImage21RenderAdvance::Rendered(rendered))
            }
            RenderPhase::Completed => Err(QwenImage21EngineError::Execution {
                description: "the render session already produced its image".to_owned(),
            }),
        }
    }

    /// Tokenize the full template, encode it, slice off the system block, release the encoder.
    fn condition_prompt(
        &mut self,
        runtime: &MlxRuntime,
    ) -> Result<MlxArray, QwenImage21EngineError> {
        let encoder = self
            .encoder
            .take()
            .ok_or_else(|| released_component_error("the text encoder"))?;
        // 1. Condition: tokenize the full chat template (system block included), encode it,
        //    then slice off the system block the transformer was never trained to read.
        let template = render_t2i_prompt_template(normalize_empty_prompt(&self.request.prompt));
        let token_ids = self
            .tokenizer
            .encode(template.as_str(), false)
            .map_err(|source| QwenImage21EngineError::Execution {
                description: format!("the prompt failed to tokenize: {source}"),
            })?
            .get_ids()
            .to_vec();
        let drop_idx = self.system_block_token_count();
        let attention_mask = vec![1_u32; token_ids.len()];
        let conditioning = build_prompt_conditioning(
            &[token_ids.clone()],
            &[attention_mask],
            drop_idx,
            self.image_pad_token_id()?,
        )
        .map_err(|source| QwenImage21EngineError::Execution {
            description: format!("the prompt failed to condition: {source}"),
        })?;
        self.vlm_image_mask = conditioning
            .image_pad_mask
            .first()
            .map(|mask| mask.iter().map(|&is_image| is_image != 0).collect())
            .unwrap_or_default();
        let hidden_states = encoder.encode(runtime, &token_ids)?;
        drop(encoder);
        let sliced_hidden = runtime.slice(
            &hidden_states,
            &[0, drop_idx as i32, 0],
            &[1, token_ids.len() as i32, TEXT_EMBEDDING_WIDTH],
            &[1, 1, 1],
        )?;
        sliced_hidden.evaluate()?;
        Ok(sliced_hidden)
    }

    /// Resolve the geometry, build the schedule, and draw the deterministic initial noise.
    fn prepare_noise(&mut self, runtime: &MlxRuntime) -> Result<(), QwenImage21EngineError> {
        let (pixel_height, pixel_width) = resolve_generation_dimensions(
            Some(self.request.height),
            Some(self.request.width),
            1024,
        );
        let (latent_height, latent_width) = latent_spatial_dimensions(pixel_height, pixel_width);
        let target_tokens = latent_height * latent_width;
        self.latent_height = latent_height;
        self.latent_width = latent_width;
        self.rendered_width = pixel_width;
        self.rendered_height = pixel_height;
        self.img_shapes = [(latent_height, latent_width)];
        let mu = default_shift(target_tokens as f64);
        let schedule = build_schedule(
            self.request.num_inference_steps,
            mu,
            &FlowMatchSchedulerParams::reviewed_artifact(),
        );
        // Deterministic initial noise, packed as `(1, tokens, 64)` latents.
        let noise_key = runtime.random_key(self.request.seed)?;
        let latents = runtime.random_normal(
            &[
                1,
                target_tokens as i32,
                QWEN_IMAGE_21_LATENT_CHANNEL_COUNT as i32,
            ],
            MlxDtype::BFloat16,
            0.0,
            1.0,
            &noise_key,
        )?;
        self.schedule = Some(schedule);
        self.latents = Some(latents);
        Ok(())
    }

    /// One flow-matching Euler step: predict, then `latents += (sigma_next - sigma) · prediction`.
    fn denoise_one_step(&mut self, runtime: &MlxRuntime) -> Result<usize, QwenImage21EngineError> {
        let schedule = self
            .schedule
            .as_ref()
            .ok_or_else(|| missing_session_state("the flow-matching schedule"))?;
        let latents = self
            .latents
            .take()
            .ok_or_else(|| missing_session_state("the packed latents"))?;
        let sliced_hidden = self
            .sliced_hidden
            .as_ref()
            .ok_or_else(|| missing_session_state("the sliced prompt hidden states"))?;
        let transformer = self
            .transformer
            .as_ref()
            .ok_or_else(|| released_component_error("the transformer"))?;
        let step_index = self.step_index;
        let sigma = schedule.transformer_sigmas()[step_index];
        let sigma_next = schedule.sigmas[step_index + 1];
        let prediction = transformer.forward(
            runtime,
            &QwenImage21TransformerRequest {
                packed_latents: &latents,
                text_embeddings: sliced_hidden,
                vlm_image_mask: &self.vlm_image_mask,
                img_shapes: &self.img_shapes,
                timesteps: &[sigma],
            },
        )?;
        let step = runtime.multiply_scalar(&prediction, sigma_next - sigma)?;
        let latents = runtime.add(&latents, &step)?;
        latents.evaluate()?;
        self.latents = Some(latents);
        self.step_index += 1;
        Ok(self.step_index)
    }

    /// Unpack the latents, release the transformer, decode, and convert to RGB bytes.
    /// Unpack the latents, release the transformer, and decode to clamped RGBA pixels.
    fn decode_latents(&mut self, runtime: &MlxRuntime) -> Result<(), QwenImage21EngineError> {
        let latents = self
            .latents
            .take()
            .ok_or_else(|| missing_session_state("the packed latents"))?;
        let packed_grid_latents = runtime.reshape(
            &latents,
            &[
                1,
                self.latent_height as i32,
                self.latent_width as i32,
                QWEN_IMAGE_21_LATENT_CHANNEL_COUNT as i32,
            ],
        )?;
        // The reference releases the transformer after the denoise loop; the VAE decode then
        // owns the working set with only the VAE's weights resident.
        drop(self.transformer.take());
        // The reference hands the VAE latents in the VAE's dtype before decoding; the
        // transformer runs BF16 while the reviewed VAE weights are F32.
        let grid_latents = runtime.astype(&packed_grid_latents, MlxDtype::Float32)?;
        let pixels = self
            .decoder
            .decode_image_latents(runtime, &grid_latents)
            .map_err(|source| match source {
                crate::qwen_image_21::QwenImage21VaeError::Mlx(inner) => {
                    QwenImage21EngineError::Mlx(inner)
                }
                other => QwenImage21EngineError::Execution {
                    description: format!("the VAE decode failed: {other}"),
                },
            })?;
        let pixel_values = pixels
            .to_vec_f32()
            .map_err(|source| QwenImage21EngineError::Mlx(source))?;
        self.pixel_values = Some(pixel_values);
        Ok(())
    }

    /// `(x + 1) / 2 · 255` per channel, clamped to bytes, first three channels only.
    fn convert_pixels(&self) -> QwenImage21Rendered {
        let pixel_values = self.pixel_values.as_deref().unwrap_or(&[]);
        QwenImage21Rendered {
            width: self.rendered_width,
            height: self.rendered_height,
            rgb_bytes: rgba_to_rgb8(pixel_values),
        }
    }

    /// The system block's token count — the reference's `_drop_idx`, derived by tokenizing the
    /// exact chat-template prefix the pipeline renders.
    fn system_block_token_count(&self) -> usize {
        let system_prefix = format!(
            "{im_start}system\n{sys}{im_end}\n",
            im_start = special_tokens::IM_START,
            sys = QWEN_IMAGE_21_SYS_PROMPT,
            im_end = special_tokens::IM_END,
        );
        self.tokenizer
            .encode(system_prefix.as_str(), false)
            .map(|encoding| encoding.get_ids().len())
            .unwrap_or(0)
    }

    fn image_pad_token_id(&self) -> Result<u32, QwenImage21EngineError> {
        self.tokenizer
            .token_to_id(special_tokens::IMAGE_PAD)
            .ok_or_else(|| QwenImage21EngineError::Execution {
                description: "the artifact tokenizer is missing the image-pad token".to_owned(),
            })
    }
}

fn released_component_error(component: &str) -> QwenImage21EngineError {
    QwenImage21EngineError::Execution {
        description: format!("{component} was released by an earlier phase of this render"),
    }
}

fn missing_session_state(state: &str) -> QwenImage21EngineError {
    QwenImage21EngineError::Execution {
        description: format!("the render session is missing {state}"),
    }
}
