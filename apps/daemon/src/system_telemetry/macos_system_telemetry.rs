#![allow(unsafe_code)]

//! macOS system telemetry: Apple GPU utilization from IOKit and memory
//! pressure from the kernel's `kern.memorystatus_vm_pressure_level` sysctl.
//!
//! This module owns the daemon's only FFI block; all other daemon modules
//! retain `deny(unsafe_code)`. The sampling logic mirrors
//! `SystemTelemetrySampler.swift`: iterate `AGXAccelerator` services, read
//! `PerformanceStatistics`, and extract `Device Utilization %` as a 0–100
//! double.

use std::time::Duration;

use tokio::{process::Command, time::timeout};

use super::SystemTelemetrySource;
use super::memory_pressure_levels::parse_macos_memory_pressure_level;

/// The macOS telemetry source.
pub(crate) struct MacOsSystemTelemetry;

impl SystemTelemetrySource for MacOsSystemTelemetry {
    fn sample_gpu_utilization_percentage() -> Option<f64> {
        sample_gpu_utilization_percentage()
    }

    async fn sample_memory_pressure_level() -> Option<&'static str> {
        sample_memory_pressure_level().await
    }
}

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
const MEMORY_PRESSURE_SYSCTL_KEY: &str = "kern.memorystatus_vm_pressure_level";
const SYSCTL_EXECUTABLE_PATH: &str = "/usr/sbin/sysctl";
const SYSCTL_SAMPLE_TIMEOUT: Duration = Duration::from_secs(2);

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

async fn sample_memory_pressure_level() -> Option<&'static str> {
    let mut sysctl_command = Command::new(SYSCTL_EXECUTABLE_PATH);
    sysctl_command
        .arg("-n")
        .arg(MEMORY_PRESSURE_SYSCTL_KEY)
        .kill_on_drop(true);
    let sysctl_output = match timeout(SYSCTL_SAMPLE_TIMEOUT, sysctl_command.output()).await {
        Ok(Ok(sysctl_output)) => sysctl_output,
        Ok(Err(sysctl_error)) => {
            tracing::debug!(%sysctl_error, "macOS memory pressure probe failed");
            return None;
        }
        Err(_) => {
            tracing::debug!("macOS memory pressure probe timed out");
            return None;
        }
    };
    if !sysctl_output.status.success() {
        tracing::debug!("macOS memory pressure probe returned a failure status");
        return None;
    }
    parse_macos_memory_pressure_level(&String::from_utf8_lossy(&sysctl_output.stdout))
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
        let accelerator_utilization_percentage = read_accelerator_utilization(accelerator_iterator);
        IOObjectRelease(accelerator_iterator);
        accelerator_utilization_percentage
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
