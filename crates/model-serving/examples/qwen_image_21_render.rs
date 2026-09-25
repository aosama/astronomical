//! Renders one Qwen-Image-2.1 image from the installed artifact.
//!
//! Usage:
//! ```text
//! ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY=<artifact root> \
//!     cargo run -p astronomical-model-serving --features direct-mlx \
//!     --example qwen_image_21_render -- --prompt "..." --output ~/Downloads/qwen.png
//! ```
//!
//! Defaults: 1024×1024, 20 denoising steps, a fixed seed, and a prompt from the repo's
//! mandated Romeo and Juliet source text. Prints live per-step progress and the final path.

use std::path::PathBuf;
use std::time::Instant;

use astronomical_model_serving::{QwenImage21Pipeline, QwenImage21RenderRequest};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

const ARTIFACT_ENV: &str = "ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY";
const DEFAULT_PROMPT: &str = "It is the east, and Juliet is the sun. Golden dawn light over a \
                              balcony garden, oil painting style.";
const DEFAULT_WIDTH: usize = 1024;
const DEFAULT_HEIGHT: usize = 1024;
const DEFAULT_STEPS: usize = 20;
const DEFAULT_SEED: u64 = 20260924;
/// Wired-memory headroom above the sum of the three components' weight files; the render
/// releases the text encoder after conditioning, so this covers the transformer's activations
/// and the VAE's full-resolution decode peak.
const ACTIVATION_HEADROOM_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// The MLX allocator cache is capped small: this one-image process has no reuse to bank on,
/// and the memory ceiling is the real limit.
const ALLOCATOR_CACHE_LIMIT_BYTES: usize = 8 * 1024 * 1024;

fn main() {
    match render() {
        Ok(()) => {}
        Err(message) => {
            eprintln!("qwen_image_21_render: {message}");
            std::process::exit(1);
        }
    }
}

fn render() -> Result<(), String> {
    let artifact_root = std::env::var_os(ARTIFACT_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| {
            format!("set {ARTIFACT_ENV} to the installed Qwen-Image-2.1 artifact root")
        })?;
    let mut arguments = std::env::args().skip(1).peekable();
    let mut prompt = DEFAULT_PROMPT.to_owned();
    let mut output_path: Option<PathBuf> = None;
    let mut width = DEFAULT_WIDTH;
    let mut height = DEFAULT_HEIGHT;
    let mut steps = DEFAULT_STEPS;
    let mut seed = DEFAULT_SEED;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--prompt" => {
                prompt = arguments
                    .next()
                    .ok_or_else(|| "--prompt requires a value".to_owned())?;
            }
            "--output" => {
                output_path = Some(PathBuf::from(
                    arguments
                        .next()
                        .ok_or_else(|| "--output requires a path".to_owned())?,
                ));
            }
            "--width" => {
                width = parse_argument(&mut arguments, "--width")?;
            }
            "--height" => {
                height = parse_argument(&mut arguments, "--height")?;
            }
            "--steps" => {
                steps = parse_argument(&mut arguments, "--steps")?;
            }
            "--seed" => {
                let seed_value = arguments
                    .next()
                    .ok_or_else(|| "--seed requires a value".to_owned())?;
                seed = seed_value
                    .parse()
                    .map_err(|_| format!("--seed expects a number, received {seed_value:?}"))?;
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    let output_path = output_path.unwrap_or_else(|| PathBuf::from("qwen_image_21_render.png"));

    let component_bytes = ["transformer", "vae", "text_encoder"]
        .iter()
        .map(|component| {
            std::fs::metadata(artifact_root.join(component).join("model.safetensors"))
                .map_err(|error| format!("the {component} weights should be readable: {error}"))
                .and_then(|metadata| {
                    usize::try_from(metadata.len())
                        .map_err(|_| format!("the {component} weights exceed the platform's usize"))
                })
        })
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .sum::<usize>();
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            component_bytes + ACTIVATION_HEADROOM_BYTES,
            ALLOCATOR_CACHE_LIMIT_BYTES,
        )
        .map_err(|error| format!("the render memory limits are invalid: {error}"))?,
    )
    .map_err(|error| format!("the MLX runtime failed to initialize: {error}"))?;

    println!(
        "qwen_image_21_render: loading the pipeline from {}",
        artifact_root.display()
    );
    let load_started = Instant::now();
    let pipeline =
        QwenImage21Pipeline::load(&runtime, &artifact_root).map_err(|error| error.to_string())?;
    println!(
        "qwen_image_21_render: pipeline loaded in {:.1}s",
        load_started.elapsed().as_secs_f32()
    );

    let request = QwenImage21RenderRequest {
        prompt,
        width,
        height,
        num_inference_steps: steps,
        seed,
    };
    println!(
        "qwen_image_21_render: rendering {}×{} in {steps} steps, seed {seed}",
        request.width, request.height
    );
    let render_started = Instant::now();
    let mut progress_reporter = |step_index: usize, total_steps: usize| {
        println!(
            "qwen_image_21_render: denoise step {}/{} ({:.1}s elapsed)",
            step_index + 1,
            total_steps,
            render_started.elapsed().as_secs_f32()
        );
    };
    let rendered = pipeline
        .render(&runtime, &request, &mut progress_reporter)
        .map_err(|error| error.to_string())?;
    let png_bytes = rendered.to_png_bytes().map_err(|error| error.to_string())?;

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("the output parent should create: {error}"))?;
    }
    std::fs::write(&output_path, &png_bytes)
        .map_err(|error| format!("the render output should write: {error}"))?;
    println!(
        "qwen_image_21_render: wrote {} ({} bytes) in {:.1}s total",
        output_path.display(),
        png_bytes.len(),
        render_started.elapsed().as_secs_f32()
    );
    Ok(())
}

fn parse_argument(
    arguments: &mut std::iter::Peekable<std::iter::Skip<std::env::Args>>,
    flag: &str,
) -> Result<usize, String> {
    let value = arguments
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))?;
    value
        .parse()
        .map_err(|_| format!("{flag} expects a number, received {value:?}"))
}
