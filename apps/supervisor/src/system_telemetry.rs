#![allow(unsafe_code)]

//! Local macOS system telemetry exposed by the supervisor.
//!
//! The Swift menu bar app samples GPU utilization directly from IOKit. A
//! local API client can request the same measurement through
//! `GET /v1/system/telemetry`. This module owns the crate's only FFI
//! block; all other supervisor modules retain `deny(unsafe_code)`.
//!
//! The sampling logic mirrors `SystemTelemetrySampler.swift`: iterate
//! `AGXAccelerator` services, read `PerformanceStatistics`, and extract
//! `Device Utilization %` as a 0–100 double.

use axum::{
    Json, Router,
    response::{IntoResponse, Response},
    routing::get,
};
use serde::Serialize;

// IOKit and CoreFoundation FFI types. These are opaque pointer types from the
// C frameworks; in Rust they are represented as `*const c_void`.
type KernReturn = i32;
type IoObjectT = u32;
type IoIteratorT = IoObjectT;
type MachPortT = u32;
type CfAllocatorRef = *const std::ffi::c_void;
type CfStringRef = *const std::ffi::c_void;
type CfTypeRef = *const std::ffi::c_void;

const KERN_SUCCESS: KernReturn = 0;
// kIOMainPortDefault is a macro (#define kIOMainPortDefault 0), not a function.
const IO_MAIN_PORT_DEFAULT: MachPortT = 0;
const K_CF_ENCODING_UTF8: u32 = 0x08000100;
const K_CF_NUMBER_FLOAT64_TYPE: i32 = 13;
const K_CF_NUMBER_SINT32_TYPE: i32 = 3;

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOServiceMatching(name: *const std::os::raw::c_char) -> *mut std::ffi::c_void;
    fn IOServiceGetMatchingServices(
        main_port: MachPortT,
        matching: *mut std::ffi::c_void,
        existing_iterator: *mut IoIteratorT,
    ) -> KernReturn;
    fn IOIteratorNext(iterator: IoIteratorT) -> IoObjectT;
    fn IOObjectRelease(object: IoObjectT) -> KernReturn;
    fn IORegistryEntryCreateCFProperty(
        entry: IoObjectT,
        property_name: CfStringRef,
        allocator: CfAllocatorRef,
        options: u32,
    ) -> CfTypeRef;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(
        allocator: CfAllocatorRef,
        c_string: *const std::os::raw::c_char,
        encoding: u32,
    ) -> CfStringRef;
    fn CFRelease(cf_object: CfTypeRef);
    fn CFGetTypeID(cf_object: CfTypeRef) -> usize;
    fn CFDictionaryGetTypeID() -> usize;
    fn CFNumberGetTypeID() -> usize;
    fn CFDictionaryGetValue(dictionary: CfTypeRef, key: CfTypeRef) -> CfTypeRef;
    /// Returns 1 on success, 0 on failure. Writes the value into `value_ptr`.
    fn CFNumberGetValue(
        number: CfTypeRef,
        number_type: i32,
        value_ptr: *mut std::ffi::c_void,
    ) -> std::ffi::c_int;
}

/// JSON body returned by `GET /v1/system/telemetry`.
#[derive(Serialize)]
pub(crate) struct SystemTelemetryDocument {
    pub gpu_utilization_percentage: Option<f64>,
}

/// `GET /v1/system/telemetry` — samples the local GPU utilization percentage
/// from the first AGX accelerator's IOKit PerformanceStatistics.
pub(crate) async fn sample_telemetry() -> Response {
    let gpu_utilization_percentage = match tokio::task::spawn_blocking(
        sample_gpu_utilization_percentage,
    )
    .await
    {
        Ok(gpu_utilization_percentage) => gpu_utilization_percentage,
        Err(telemetry_task_error) => {
            tracing::debug!(%telemetry_task_error, "GPU utilization sample task did not complete");
            None
        }
    };
    Json(SystemTelemetryDocument {
        gpu_utilization_percentage,
    })
    .into_response()
}

/// Returns the system-telemetry route, ready to merge into any supervisor
/// `Router`. Keeps `application.rs` from growing past the 500-line principle.
pub(crate) fn system_telemetry_routes<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new().route("/v1/system/telemetry", get(sample_telemetry))
}

/// Reads the Apple GPU `Device Utilization %` from IOKit. Returns `None` when
/// no AGX accelerator is found or the statistic is absent or out of range.
fn sample_gpu_utilization_percentage() -> Option<f64> {
    // SAFETY: IOServiceMatching returns a CFDictionary that
    // IOServiceGetMatchingServices consumes and releases. The iterator handle
    // is released before returning. Each accelerator service handle is
    // released in the loop body. All CFStringRef and CFTypeRef values created
    // here are released before returning.
    unsafe {
        let matching_dictionary = IOServiceMatching(c"AGXAccelerator".as_ptr());
        if matching_dictionary.is_null() {
            return None;
        }
        let mut accelerator_iterator: IoIteratorT = 0;
        let matching_result = IOServiceGetMatchingServices(
            IO_MAIN_PORT_DEFAULT,
            matching_dictionary,
            &mut accelerator_iterator as *mut IoIteratorT,
        );
        if matching_result != KERN_SUCCESS {
            return None;
        }
        let result = read_accelerator_utilization(accelerator_iterator);
        IOObjectRelease(accelerator_iterator);
        result
    }
}

fn read_accelerator_utilization(accelerator_iterator: IoIteratorT) -> Option<f64> {
    // SAFETY: `accelerator_iterator` comes from IOServiceGetMatchingServices.
    // Every service and Core Foundation object acquired below is released on
    // every path before the next service or return.
    unsafe {
        let performance_statistics_name = CFStringCreateWithCString(
            std::ptr::null(),
            c"PerformanceStatistics".as_ptr(),
            K_CF_ENCODING_UTF8,
        );
        if performance_statistics_name.is_null() {
            return None;
        }
        let utilization_name = CFStringCreateWithCString(
            std::ptr::null(),
            c"Device Utilization %".as_ptr(),
            K_CF_ENCODING_UTF8,
        );
        if utilization_name.is_null() {
            CFRelease(performance_statistics_name);
            return None;
        }

        let mut utilization_percentage = None;
        loop {
            let accelerator_service = IOIteratorNext(accelerator_iterator);
            if accelerator_service == 0 {
                break;
            }
            let performance_statistics = IORegistryEntryCreateCFProperty(
                accelerator_service,
                performance_statistics_name,
                std::ptr::null(),
                0,
            );
            IOObjectRelease(accelerator_service);
            if performance_statistics.is_null()
                || CFGetTypeID(performance_statistics) != CFDictionaryGetTypeID()
            {
                if !performance_statistics.is_null() {
                    CFRelease(performance_statistics);
                }
                continue;
            }
            let utilization_number = CFDictionaryGetValue(performance_statistics, utilization_name);
            if !utilization_number.is_null()
                && CFGetTypeID(utilization_number) == CFNumberGetTypeID()
            {
                let mut floating_point_utilization_percentage = 0.0;
                let floating_point_conversion_succeeded = CFNumberGetValue(
                    utilization_number,
                    K_CF_NUMBER_FLOAT64_TYPE,
                    &mut floating_point_utilization_percentage as *mut f64 as *mut std::ffi::c_void,
                );
                if floating_point_conversion_succeeded != 0
                    && (0.0..=100.0).contains(&floating_point_utilization_percentage)
                {
                    utilization_percentage = Some(floating_point_utilization_percentage);
                } else {
                    let mut integer_utilization_percentage = 0;
                    let integer_conversion_succeeded = CFNumberGetValue(
                        utilization_number,
                        K_CF_NUMBER_SINT32_TYPE,
                        &mut integer_utilization_percentage as *mut i32 as *mut std::ffi::c_void,
                    );
                    if integer_conversion_succeeded != 0
                        && (0..=100).contains(&integer_utilization_percentage)
                    {
                        utilization_percentage = Some(f64::from(integer_utilization_percentage));
                    }
                }
            }
            CFRelease(performance_statistics);
            if utilization_percentage.is_some() {
                break;
            }
        }
        CFRelease(utilization_name);
        CFRelease(performance_statistics_name);
        utilization_percentage
    }
}
