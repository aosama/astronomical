//! Compiles the Objective-C Core ML bridge and links the Apple frameworks it
//! needs.
//!
//! The bridge is Objective-C rather than Swift because Core ML's full
//! Objective-C surface (compute plans, IOSurface-backed multi-arrays,
//! prediction options) is reachable from plain `clang` through the `cc`
//! crate, so the crate builds with Cargo alone: no Swift package, no Xcode
//! project, no extra toolchain step.

fn main() {
    println!("cargo:rerun-if-changed=native/core_ml_bridge.m");
    println!("cargo:rerun-if-changed=native/system_wired_memory.c");
    cc::Build::new()
        .file("native/core_ml_bridge.m")
        .file("native/system_wired_memory.c")
        .flag("-fobjc-arc")
        .flag("-fmodules")
        // The worker links this bridge and supports macOS 14.0, so anything
        // newer stays weak: MLComputePlan (14.4) is guarded at run time, and an
        // ML Program this OS cannot compile fails its load instead of the
        // worker failing to launch.
        .flag("-mmacosx-version-min=14.0")
        .warnings_into_errors(true)
        .compile("astronomical_core_ml_bridge");
    for framework in ["CoreML", "Foundation", "CoreVideo", "IOSurface"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
}
