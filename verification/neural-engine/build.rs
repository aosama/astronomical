//! Compiles the process-memory readings and links Accelerate for the fp32
//! reference. Both live here so that serving, which needs neither, does not link
//! them.

fn main() {
    println!("cargo:rerun-if-changed=native/process_memory.c");
    println!("cargo:rerun-if-changed=native/process_memory_linux.c");
    // The readings come from Mach on macOS and from /proc on Linux; the fp32
    // reference uses Accelerate on macOS and a plain loop elsewhere, so only
    // macOS links a framework.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .file("native/process_memory.c")
            .flag("-mmacosx-version-min=14.0")
            .warnings_into_errors(true)
            .compile("astronomical_process_memory");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    } else {
        cc::Build::new()
            .file("native/process_memory_linux.c")
            .warnings_into_errors(true)
            .compile("astronomical_process_memory");
    }
}
