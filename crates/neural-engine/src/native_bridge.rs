//! Raw declarations of the Objective-C Core ML bridge and Accelerate's BLAS.
//!
//! Only the safe owners in this crate call these. The one raw pointer that
//! leaves the crate is `LockedOutputSurface`'s, because handing a surface to
//! MLX without a copy is the point of that guard.

use std::ffi::{c_char, c_void};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeComputeDeviceReport {
    pub neural_engine_count: i32,
    pub neural_engine_core_count: i32,
    pub gpu_count: i32,
    pub cpu_count: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeLayerPlacement {
    pub preferred_placement: i32,
    pub neural_engine_supported: i32,
    pub gpu_supported: i32,
    pub cpu_supported: i32,
    pub operations_off_neural_engine: i32,
}

/// Opaque owner of a loaded Core ML projection and its IOSurfaces.
#[repr(C)]
pub struct NativeCoreMlProjection {
    _opaque: [u8; 0],
}

pub const CBLAS_ROW_MAJOR: i32 = 101;
pub const CBLAS_NO_TRANSPOSE: i32 = 111;

unsafe extern "C" {
    pub fn astronomical_core_ml_report_compute_devices(
        report: *mut NativeComputeDeviceReport,
    ) -> i32;

    pub fn astronomical_core_ml_projection_load(
        model_path: *const c_char,
        compute_units_code: i32,
        input_channels: i32,
        output_channels: i32,
        token_count: i32,
        error_message: *mut c_char,
        error_capacity: u32,
    ) -> *mut NativeCoreMlProjection;

    pub fn astronomical_core_ml_projection_placement(
        projection: *const NativeCoreMlProjection,
    ) -> NativeLayerPlacement;

    pub fn astronomical_core_ml_projection_write_input(
        projection: *mut NativeCoreMlProjection,
        fp16_activations: *const u16,
        element_count: u64,
    ) -> i32;

    pub fn astronomical_core_ml_projection_read_output(
        projection: *const NativeCoreMlProjection,
        fp16_projected: *mut u16,
        element_count: u64,
    ) -> i32;

    pub fn astronomical_core_ml_projection_lock_output(
        projection: *const NativeCoreMlProjection,
        base_address: *mut *const c_void,
        bytes_per_row: *mut u64,
    ) -> i32;

    pub fn astronomical_core_ml_projection_unlock_output(projection: *const NativeCoreMlProjection);

    pub fn astronomical_core_ml_projection_predict(
        projection: *mut NativeCoreMlProjection,
        elapsed_nanoseconds: *mut u64,
        error_message: *mut c_char,
        error_capacity: u32,
    ) -> i32;

    pub fn astronomical_core_ml_projection_begin_predict(
        projection: *mut NativeCoreMlProjection,
    ) -> i32;

    pub fn astronomical_core_ml_projection_finish_predict(
        projection: *mut NativeCoreMlProjection,
        timeout_nanoseconds: u64,
        elapsed_nanoseconds: *mut u64,
    ) -> i32;

    pub fn astronomical_core_ml_projection_discard_compiled_model(
        projection: *mut NativeCoreMlProjection,
    );
    pub fn astronomical_core_ml_projection_release(projection: *mut NativeCoreMlProjection);

    // Accelerate's CBLAS single-precision matrix product (BLAS level 3).
    #[allow(clippy::too_many_arguments)]
    pub fn cblas_sgemm(
        order: i32,
        transpose_a: i32,
        transpose_b: i32,
        row_count: i32,
        column_count: i32,
        inner_count: i32,
        alpha: f32,
        matrix_a: *const f32,
        leading_dimension_a: i32,
        matrix_b: *const f32,
        leading_dimension_b: i32,
        beta: f32,
        matrix_c: *mut f32,
        leading_dimension_c: i32,
    );

    // native/process_memory.c: zero when the kernel query fails.
    pub fn astronomical_process_physical_footprint_bytes() -> u64;
    pub fn astronomical_system_wired_bytes() -> u64;
    pub fn astronomical_physical_memory_bytes() -> u64;
}
