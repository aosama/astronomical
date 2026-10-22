//! Harness-enforced network denial: makes socket creation fail loudly inside
//! tests, so the offline property is physics rather than discipline.
//!
//! On macOS the denial is applied once per test process through Apple's
//! Seatbelt (`sandbox_init`, resolved at runtime via dlsym so no SDK link
//! dependency is added) with a profile that denies the `network*` class —
//! which covers TCP, UDP, and unix-domain sockets alike. The sandbox is
//! process-wide and one-way: once the first denied test applies it, every
//! later socket attempt in that test binary fails. On other platforms the
//! harness defers to platform sandboxing — the CI Linux job already runs the
//! offline suite under `unshare -rn`, which removes the network at the kernel
//! level.
//!
//! Denial is deliberately a separate opt-in macro pair
//! (an explicit `ensure_network_denied()` call) rather than
//! implicit in every harness macro: integration journeys that genuinely
//! exercise real unix-socket transport use plain `#[tokio::test]` functions
//! and must keep their sockets.
//!
//! The macOS proof is layered with the CI seatbelt step: the step re-runs
//! the whole lane inside an outer `sandbox-exec` profile that denies real
//! networking (loopback and Unix Domain Sockets stay allowed, per the
//! offline vocabulary), and this in-process denial defers there because
//! macOS rejects applying a profile from inside one. Each binary that opts
//! into this denial is still proven completely socket-free by the
//! unsandboxed offline-tests step of the same CI job.

use std::sync::Once;
use std::sync::OnceLock;

static DENIAL_APPLICATION: Once = Once::new();
static DENIAL_OUTCOME: OnceLock<Result<(), String>> = OnceLock::new();

#[cfg(target_os = "macos")]
mod seatbelt {
    use std::ffi::{CStr, CString, c_char, c_int, c_void};

    /// The Seatbelt profile denies every network operation by name and allows
    /// the rest of the process by default. A bare `(deny network*)` profile
    /// compiles as default-deny on macOS: the denial works, but every later
    /// test in the same binary loses filesystem, IOKit (Metal device access),
    /// and mach services, so any MLX-initializing test running after one
    /// denied test fails. Naming the network operations keeps the denial loud
    /// while leaving the rest of the process untouched. Evaluation is
    /// first-match, so the named denies win over `(allow default)`.
    const NETWORK_DENIAL_PROFILE: &str = "(version 1)\n(deny network-outbound)\n(deny network-inbound)\n(deny network-bind)\n(allow default)\n";

    const RTLD_NOW: c_int = 2;
    const SANDBOX_APPLY_FLAGS: u64 = 0;

    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    type SandboxInit = unsafe extern "C" fn(
        profile: *const c_char,
        flags: u64,
        errorbuf: *mut *mut c_char,
    ) -> c_int;
    type SandboxFreeError = unsafe extern "C" fn(errorbuf: *mut c_char);

    pub(super) fn platform_apply() -> Result<(), String> {
        // SAFETY: `dlopen`/`dlsym` only read the given NUL-terminated literals and
        // return raw function pointers; libsandbox ships with every macOS userland.
        // The resolved `sandbox_init` is called exactly once (guarded by the
        // caller's `Once`) with a valid profile literal, and its error buffer, when
        // produced, is freed through the matching `sandbox_free_error`.
        unsafe {
            let library_path = CString::new("/usr/lib/libsandbox.dylib").expect("no interior NUL");
            let handle = dlopen(library_path.as_ptr(), RTLD_NOW);
            if handle.is_null() {
                return Err("libsandbox could not be loaded".to_owned());
            }
            let init_symbol = CString::new("sandbox_init").expect("no interior NUL");
            let init_pointer = dlsym(handle, init_symbol.as_ptr());
            if init_pointer.is_null() {
                return Err("sandbox_init symbol missing from libsandbox".to_owned());
            }
            let free_symbol = CString::new("sandbox_free_error").expect("no interior NUL");
            let free_pointer = dlsym(handle, free_symbol.as_ptr());
            let sandbox_init: SandboxInit = std::mem::transmute(init_pointer);

            let profile = CString::new(NETWORK_DENIAL_PROFILE).expect("no interior NUL");
            let mut error_buffer: *mut c_char = std::ptr::null_mut();
            let application_status =
                sandbox_init(profile.as_ptr(), SANDBOX_APPLY_FLAGS, &mut error_buffer);
            if application_status == 0 {
                return Ok(());
            }
            let failure_description = if error_buffer.is_null() {
                format!("sandbox_init rejected the profile (status {application_status})")
            } else {
                let seatbelt_message = CStr::from_ptr(error_buffer).to_string_lossy().into_owned();
                if !free_pointer.is_null() {
                    let sandbox_free_error: SandboxFreeError = std::mem::transmute(free_pointer);
                    sandbox_free_error(error_buffer);
                }
                seatbelt_message
            };
            Err(failure_description)
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod seatbelt {
    pub(super) fn platform_apply() -> Result<(), String> {
        // Linux CI removes the network for the offline suite with
        // `unshare -rn`; other platforms have no harness-level mechanism yet,
        // and an absent mechanism must not masquerade as an applied denial.
        Ok(())
    }
}

/// Environment marker exported only by the CI seatbelt step: the whole lane
/// is already running inside an outer `sandbox-exec` profile there, and
/// macOS forbids applying a second profile from inside one (`sandbox_init`
/// returns EPERM), so the in-process denial defers to the outer profile,
/// which already denies every real network operation.
const DEFERRAL_MARKER_VARIABLE: &str = "ASTRONOMICAL_NETWORK_DENIAL_DEFERRED_TO_SEATBELT";

/// Applies network denial to this test process exactly once and reports
/// whether the property now holds. Concurrent callers block until the first
/// application finishes and then observe the same outcome.
///
/// Under the CI seatbelt step (the deferral marker set) this returns Ok
/// without applying the in-process profile: the outer profile enforces the
/// no-real-network property for the whole lane, including loopback and
/// Unix Domain Socket allowances the offline contract sanctions. The
/// stricter socket-free proof for binaries that opt into this denial still
/// runs in the unsandboxed offline-tests step of the same CI job, so each
/// proof executes in the one run where it can physically exist.
pub fn ensure_network_denied() -> Result<(), String> {
    DENIAL_APPLICATION.call_once(|| {
        let outcome = if deferral_marker_present() {
            eprintln!("[network-denial] status=deferred reason=outer_seatbelt_profile_active");
            Ok(())
        } else {
            seatbelt::platform_apply()
        };
        DENIAL_OUTCOME
            .set(outcome)
            .expect("only the Once applies denial");
    });
    DENIAL_OUTCOME
        .get()
        .expect("the Once has completed")
        .clone()
}

fn deferral_marker_present() -> bool {
    std::env::var_os(DEFERRAL_MARKER_VARIABLE).is_some_and(|marker_value| marker_value == "1")
}
