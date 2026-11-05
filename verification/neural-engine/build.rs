//! Compiles the process-memory readings and links Accelerate for the fp32
//! reference. Both live here so that serving, which needs neither, does not link
//! them.

fn main() {
    println!("cargo:rerun-if-changed=native/process_memory.c");
    cc::Build::new()
        .file("native/process_memory.c")
        .flag("-mmacosx-version-min=14.0")
        .warnings_into_errors(true)
        .compile("astronomical_process_memory");
    println!("cargo:rustc-link-lib=framework=Accelerate");
}
