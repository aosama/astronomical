//! Host-side Core ML loader for the expert-route predictor.
//!
//! Weights and compiled programs stay off the MLX budget. Load or predict
//! failures return `None` so decode keeps the CPU predictor.

use std::ffi::{CString, c_char, c_float, c_int, c_uint};
use std::path::Path;
use std::ptr::NonNull;

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
}

/// Loaded Core ML predictor. Compute units were CPU_AND_NE at load.
pub struct PredictorAneEngine {
    handle: NonNull<AstronomicalPredictorAne>,
    convolutions_on_neural_engine: bool,
}

unsafe impl Send for PredictorAneEngine {}

impl PredictorAneEngine {
    /// Loads a `.mlpackage` or compiled model. `None` on any failure.
    #[must_use]
    pub fn try_load(model_path: &Path) -> Option<Self> {
        let model_path = CString::new(model_path.to_string_lossy().as_ref()).ok()?;
        let mut error_message = [0_i8; 256];
        let handle = unsafe {
            astronomical_predictor_ane_load(
                model_path.as_ptr(),
                error_message.as_mut_ptr(),
                error_message.len() as c_uint,
            )
        };
        let handle = NonNull::new(handle)?;
        let convolutions_on_neural_engine = unsafe {
            astronomical_predictor_ane_convolutions_on_neural_engine(handle.as_ptr()) != 0
        };
        Some(Self {
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
}

impl Drop for PredictorAneEngine {
    fn drop(&mut self) {
        unsafe { astronomical_predictor_ane_free(self.handle.as_ptr()) }
    }
}
