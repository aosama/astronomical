//! Flow-matching scheduler helpers for Qwen-Image-2.1.
//!
//! The denoising schedule is a flow-matching Euler schedule whose sigmas are exponentially shifted
//! by a dynamic `mu` derived from the image sequence length, so higher-resolution generations
//! (more latent tokens) spend relatively more steps at high noise. `mu` comes from
//! `calculate_shift`, a linear map of the latent token count onto `[base_shift, max_shift]` over
//! `[base_seq_len, max_seq_len]`. The reviewed artifact's `scheduler_config.json` additionally sets
//! `shift_terminal = 0.02`, so after the dynamic shift the schedule is stretched to end one step
//! above `0` at sigma `shift_terminal` before the terminal `0.0` is appended. This module holds
//! the pure, hermetically-testable schedule math — the dynamic shift, the sigma/timestep schedule
//! with the terminal stretch, and the Euler step — that drives the denoise loop; the tensor work
//! that consumes it lands with the GPU runtime.

/// Reviewed artifact `base_image_seq_len` (latent tokens) for the dynamic shift.
pub const DEFAULT_BASE_SEQ_LEN: f64 = 256.0;
/// Reviewed artifact `max_image_seq_len` (latent tokens) for the dynamic shift.
pub const DEFAULT_MAX_SEQ_LEN: f64 = 8192.0;
/// Reviewed artifact `base_shift` at `base_seq_len`.
pub const DEFAULT_BASE_SHIFT: f64 = 0.5;
/// Reviewed artifact `max_shift` at `max_seq_len`.
pub const DEFAULT_MAX_SHIFT: f64 = 0.9;
/// Reviewed artifact `shift_terminal`: the sigma the stretched schedule must end at before the
/// appended terminal `0.0`. `0.0` disables the stretch, matching the reference's truthiness check.
pub const DEFAULT_SHIFT_TERMINAL: f64 = 0.02;

/// Diffusion timesteps the checkpoint was trained on; the sigma -> timestep scale.
pub const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

/// The scheduler constants the denoise schedule is built from.
///
/// The engine reads these from the validated artifact's `scheduler/scheduler_config.json`; the
/// defaults here are the reviewed `mlx-community/Qwen-Image-2.1-MLX-4bit` values and are what the
/// hermetic oracle is generated from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowMatchSchedulerParams {
    pub base_seq_len: f64,
    pub max_seq_len: f64,
    pub base_shift: f64,
    pub max_shift: f64,
    pub shift_terminal: f64,
}

impl FlowMatchSchedulerParams {
    /// The reviewed artifact's scheduler constants.
    #[must_use]
    pub const fn reviewed_artifact() -> Self {
        Self {
            base_seq_len: DEFAULT_BASE_SEQ_LEN,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            base_shift: DEFAULT_BASE_SHIFT,
            max_shift: DEFAULT_MAX_SHIFT,
            shift_terminal: DEFAULT_SHIFT_TERMINAL,
        }
    }
}

/// Compute the dynamic flow-matching shift `mu` for an image sequence length.
///
/// Mirrors diffusers `calculate_shift` exactly: a straight line through
/// `(base_seq_len, base_shift)` and `(max_seq_len, max_shift)`, evaluated at `image_seq_len`. The
/// scheduler then uses `mu` to exponentially shift the sigma schedule. Values outside
/// `[base_seq_len, max_seq_len]` extrapolate along the same line, matching the reference.
#[must_use]
pub fn calculate_shift(
    image_seq_len: f64,
    base_seq_len: f64,
    max_seq_len: f64,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    let slope = (max_shift - base_shift) / (max_seq_len - base_seq_len);
    let intercept = base_shift - slope * base_seq_len;
    image_seq_len * slope + intercept
}

/// `calculate_shift` with the reviewed artifact's scheduler constants.
#[must_use]
pub fn default_shift(image_seq_len: f64) -> f64 {
    let params = FlowMatchSchedulerParams::reviewed_artifact();
    calculate_shift(
        image_seq_len,
        params.base_seq_len,
        params.max_seq_len,
        params.base_shift,
        params.max_shift,
    )
}

/// Exponential dynamic time-shift, matching the reviewed config
/// (`time_shift_type="exponential"`, `use_dynamic_shifting=True`):
/// `exp(mu) / (exp(mu) + (1/t - 1))`, evaluated per sigma.
///
/// Dtype flow mirrors the reference exactly: the sigma ramp arrives as a float32 array, and
/// under numpy/torch promotion rules the `math.exp(mu)` Python float is cast to float32 before
/// the arithmetic — so the whole shift (including `exp(mu)`) is float32 math. Computing `exp(mu)`
/// in f64 and keeping the division in f64 looks more precise but is *wrong*: it lands 1 ULP off
/// the reference on the fixture vectors, which the sigma tests catch.
#[must_use]
fn time_shift_exponential_f32(sigmas: &[f32], mu: f64) -> Vec<f32> {
    let exp_mu = mu.exp() as f32;
    sigmas
        .iter()
        .map(|&sigma| {
            let shifted = 1.0f32 / sigma - 1.0f32;
            exp_mu / (exp_mu + shifted)
        })
        .collect()
}

/// `stretch_shift_to_terminal`: rescale the shifted schedule so its last entry lands at
/// `shift_terminal` instead of at the raw shifted value.
///
/// Mirrors the reference: `one_minus_z = 1 - t`; `scale_factor = one_minus_z[-1] / (1 -
/// shift_terminal)`; `t = 1 - one_minus_z / scale_factor`. The reference operates on a float32
/// array, so all of this is float32 math; `(1 - shift_terminal)` is a Python float in the
/// reference and is cast to float32 for the division.
fn stretch_shift_to_terminal(sigmas: &[f32], shift_terminal: f64) -> Vec<f32> {
    let one_minus_z: Vec<f32> = sigmas.iter().map(|&sigma| 1.0f32 - sigma).collect();
    let terminal_factor = (1.0 - shift_terminal) as f32;
    let scale_factor = *one_minus_z.last().unwrap() / terminal_factor;
    one_minus_z
        .iter()
        .map(|&value| 1.0f32 - value / scale_factor)
        .collect()
}

/// A constructed flow-matching sampling schedule.
pub struct FlowMatchSchedule {
    /// `num_inference_steps + 1` sigmas: the per-step sigmas followed by the terminal `0.0` that
    /// the reference `set_timesteps` appends. `sigmas[i]` and `sigmas[i + 1]` bound step `i`.
    pub sigmas: Vec<f32>,
    /// `num_inference_steps` timesteps = `sigma * NUM_TRAIN_TIMESTEPS`. These are the values
    /// iterated in the denoise loop; the pipeline hands the DiT `timestep / 1000` (a sigma in
    /// `[0, 1]`).
    pub timesteps: Vec<f32>,
}

impl FlowMatchSchedule {
    /// The per-step time input the denoising transformer consumes: one normalized sigma per
    /// denoising step, aligned with `sigmas` so step `i` pairs `time[i]` with the Euler bounds
    /// `sigmas[i]..sigmas[i + 1]`.
    ///
    /// The reference hands the DiT `timestep / 1000` — the sigma itself, in `[0, 1]` — while
    /// `timesteps` holds the training-timestep scale (sigma × 1000) that the scheduler reports.
    /// Pairing the loop with `timesteps` instead scales the transformer's time embedding by 1000,
    /// which still produces a plausible-looking latent grid but renders as noise. Routing the loop
    /// through this accessor keeps the convention in one place, next to the array it indexes.
    #[must_use]
    pub fn transformer_sigmas(&self) -> &[f32] {
        &self.sigmas[..self.timesteps.len()]
    }
}

/// Build the flow-matching sigma/timestep schedule for `num_inference_steps` steps at dynamic
/// shift `mu`, applying the reviewed artifact's terminal stretch.
///
/// Reproduces the Qwen `set_timesteps` path: sigmas start at the linear ramp `linspace(1.0,
/// 1/N, N)` (numpy linspace in f64 with the endpoint pinned, then cast to f32 as the reference
/// does), are exponentially time-shifted by `mu` in f32, stretched to end at
/// `params.shift_terminal` when it is set (f32), and scaled to timesteps in f32; the terminal
/// `0.0` sigma is appended last.
#[must_use]
pub fn build_schedule(
    num_inference_steps: usize,
    mu: f64,
    params: &FlowMatchSchedulerParams,
) -> FlowMatchSchedule {
    assert!(
        num_inference_steps >= 1,
        "the schedule needs at least one step"
    );
    let start = 1.0f64;
    let stop = 1.0f64 / num_inference_steps as f64;
    let step = (stop - start) / (num_inference_steps - 1) as f64;

    // Linear ramp in f64 exactly as numpy linspace computes it (`start + index * step`, endpoint
    // pinned to `stop`), cast to f32 like the reference's `.astype(np.float32)`.
    let mut ramp: Vec<f32> = (0..num_inference_steps)
        .map(|index| (start + step * index as f64) as f32)
        .collect();
    ramp[num_inference_steps - 1] = stop as f32;

    let shifted = time_shift_exponential_f32(&ramp, mu);
    let stretched = if params.shift_terminal != 0.0 {
        stretch_shift_to_terminal(&shifted, params.shift_terminal)
    } else {
        shifted
    };
    let timesteps: Vec<f32> = stretched
        .iter()
        .map(|&sigma| sigma * NUM_TRAIN_TIMESTEPS)
        .collect();

    let mut sigmas = stretched;
    sigmas.push(0.0f32); // terminal sigma the reference appends so the final step lands on the clean image

    FlowMatchSchedule { sigmas, timesteps }
}

/// One flow-matching Euler step: `prev = sample + (sigma_next - sigma) * model_output`.
///
/// Operates in f32 (the reference upcasts `sample` to f32 before the update). The caller casts the
/// result back to the model dtype at the runtime boundary; with an f32 `model_output` no cast
/// occurs, which is why this is independently testable against the float oracle.
#[must_use]
pub fn euler_step(sample: &[f32], model_output: &[f32], sigma: f32, sigma_next: f32) -> Vec<f32> {
    assert_eq!(
        sample.len(),
        model_output.len(),
        "sample and model_output widths must match"
    );
    let dt = sigma_next - sigma;
    sample
        .iter()
        .zip(model_output.iter())
        .map(|(&s, &m)| s + dt * m)
        .collect()
}
