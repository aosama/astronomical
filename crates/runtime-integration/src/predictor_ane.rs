//! Host-side Core ML loader for the expert-route predictor.
//!
//! Weights and compiled programs stay off the MLX budget. Load or predict
//! failures return `None` so decode keeps the CPU predictor.

use std::ffi::{CStr, CString, c_char, c_float, c_int, c_uint};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::ptr::NonNull;

use super::predictor_ane_export::{PredictorAneConvolutionSnapshot, write_predictor_mlmodel};

#[repr(C)]
struct AstronomicalPredictorAne {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn astronomical_predictor_ane_load(
        model_path: *const c_char,
        error_message: *mut c_char,
        error_message_capacity: c_uint,
    ) -> *mut AstronomicalPredictorAne;
    fn astronomical_predictor_ane_convolutions_on_neural_engine(
        handle: *const AstronomicalPredictorAne,
    ) -> c_int;
    fn astronomical_predictor_ane_predict(
        handle: *mut AstronomicalPredictorAne,
        head_inputs: *const c_float,
        layer_count: c_int,
        input_dim: c_int,
        logits_out: *mut c_float,
        expert_count: c_int,
    ) -> c_int;
    fn astronomical_predictor_ane_free(handle: *mut AstronomicalPredictorAne);
    fn astronomical_predictor_ane_begin(
        handle: *mut AstronomicalPredictorAne,
        head_inputs: *const c_float,
        layer_count: c_int,
        input_dim: c_int,
        expert_count: c_int,
    ) -> c_int;
    fn astronomical_predictor_ane_harvest(
        handle: *mut AstronomicalPredictorAne,
        logits_out: *mut c_float,
        logit_count: c_int,
        timeout_nanoseconds: u64,
        elapsed_nanoseconds: *mut u64,
    ) -> c_int;
}

/// Loaded Core ML predictor. Compute units were CPU_AND_NE at load.
pub struct PredictorAneEngine {
    handle: NonNull<AstronomicalPredictorAne>,
    convolutions_on_neural_engine: bool,
}

unsafe impl Send for PredictorAneEngine {}

impl PredictorAneEngine {
    /// Writes a NeuralNetwork snapshot and loads it with CPU_AND_NE.
    #[must_use]
    pub fn try_from_convolution_snapshot(
        snapshot: &PredictorAneConvolutionSnapshot,
    ) -> Option<Self> {
        let process_id = std::process::id();
        let mlmodel_path = std::env::temp_dir().join(format!(
            "astronomical-predictor-{}-{}x{}.mlmodel",
            process_id, snapshot.layer_count, snapshot.expert_count
        ));
        write_predictor_mlmodel(snapshot, &mlmodel_path).ok()?;
        let engine = Self::try_load(&mlmodel_path).ok()?;
        let warmup_inputs = vec![0.0_f32; snapshot.grouped_input_channels()];
        let _ = engine.predict(
            &warmup_inputs,
            snapshot.layer_count,
            snapshot.input_dim,
            snapshot.expert_count,
        );
        Some(engine)
    }

    /// Loads a `.mlmodel` (compiled first) or a `.mlmodelc` bundle.
    pub fn try_load(model_path: &Path) -> Result<Self, String> {
        let compiled_path =
            if model_path.extension().and_then(|ext| ext.to_str()) == Some("mlmodel") {
                compile_mlmodel(model_path)?
            } else {
                model_path.to_path_buf()
            };
        let model_path = CString::new(compiled_path.to_string_lossy().as_ref())
            .map_err(|_| "predictor Core ML path is not a C string".to_owned())?;
        let mut error_message = [0_i8; 256];
        let handle = unsafe {
            astronomical_predictor_ane_load(
                model_path.as_ptr(),
                error_message.as_mut_ptr(),
                error_message.len() as c_uint,
            )
        };
        let Some(handle) = NonNull::new(handle) else {
            let message = unsafe { CStr::from_ptr(error_message.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(message);
        };
        let convolutions_on_neural_engine = unsafe {
            astronomical_predictor_ane_convolutions_on_neural_engine(handle.as_ptr()) != 0
        };
        Ok(Self {
            handle,
            convolutions_on_neural_engine,
        })
    }

    #[must_use]
    pub fn convolutions_on_neural_engine(&self) -> bool {
        self.convolutions_on_neural_engine
    }

    /// Runs one token. `head_inputs` is layer-major `[layer][input_dim]`.
    pub fn predict(
        &self,
        head_inputs: &[f32],
        layer_count: usize,
        input_dim: usize,
        expert_count: usize,
    ) -> Option<Vec<f32>> {
        let expected_input = layer_count.checked_mul(input_dim)?;
        let expected_output = layer_count.checked_mul(expert_count)?;
        if head_inputs.len() != expected_input {
            return None;
        }
        let mut logits = vec![0.0_f32; expected_output];
        let status = unsafe {
            astronomical_predictor_ane_predict(
                self.handle.as_ptr(),
                head_inputs.as_ptr(),
                c_int::try_from(layer_count).ok()?,
                c_int::try_from(input_dim).ok()?,
                logits.as_mut_ptr(),
                c_int::try_from(expert_count).ok()?,
            )
        };
        if status != 0 {
            return None;
        }
        Some(logits)
    }

    /// Starts a Core ML predict without waiting. Fail-open if one is in flight.
    pub fn begin_predict(
        &self,
        head_inputs: &[f32],
        layer_count: usize,
        input_dim: usize,
        expert_count: usize,
    ) -> bool {
        let expected_input = layer_count.saturating_mul(input_dim);
        if head_inputs.len() != expected_input {
            return false;
        }
        let Ok(layer_count) = c_int::try_from(layer_count) else {
            return false;
        };
        let Ok(input_dim) = c_int::try_from(input_dim) else {
            return false;
        };
        let Ok(expert_count) = c_int::try_from(expert_count) else {
            return false;
        };
        unsafe {
            astronomical_predictor_ane_begin(
                self.handle.as_ptr(),
                head_inputs.as_ptr(),
                layer_count,
                input_dim,
                expert_count,
            ) == 0
        }
    }

    /// Waits up to `timeout_nanoseconds` for `begin_predict`. Returns logits and
    /// engine-side elapsed nanoseconds.
    pub fn harvest_predict(
        &self,
        layer_count: usize,
        expert_count: usize,
        timeout_nanoseconds: u64,
    ) -> Option<(Vec<f32>, u64)> {
        let logit_count = layer_count.checked_mul(expert_count)?;
        let mut logits = vec![0.0_f32; logit_count];
        let mut elapsed_nanoseconds = 0_u64;
        let status = unsafe {
            astronomical_predictor_ane_harvest(
                self.handle.as_ptr(),
                logits.as_mut_ptr(),
                c_int::try_from(logit_count).ok()?,
                timeout_nanoseconds,
                &mut elapsed_nanoseconds,
            )
        };
        if status != 0 {
            return None;
        }
        Some((logits, elapsed_nanoseconds))
    }
}

impl Drop for PredictorAneEngine {
    fn drop(&mut self) {
        unsafe { astronomical_predictor_ane_free(self.handle.as_ptr()) }
    }
}

fn compile_mlmodel(mlmodel_path: &Path) -> Result<PathBuf, String> {
    let parent_directory = mlmodel_path
        .parent()
        .ok_or_else(|| "predictor mlmodel path has no parent".to_owned())?;
    let compiled_path = parent_directory.join(format!(
        "{}.mlmodelc",
        mlmodel_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| "predictor mlmodel stem is missing".to_owned())?
    ));
    let compile_status = Command::new("xcrun")
        .args(["coremlcompiler", "compile"])
        .arg(mlmodel_path)
        .arg(parent_directory)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|error| format!("coremlcompiler failed to start: {error}"))?;
    if !compile_status.success() || !compiled_path.exists() {
        return Err("coremlcompiler did not produce a compiled predictor".to_owned());
    }
    Ok(compiled_path)
}
