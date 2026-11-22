//! Compiles the native record reader.
//!
//! The reader is C++ and not Rust because its whole job is the wait between
//! a batch being handed over and its threads issuing reads: it owns its
//! threads, their wake-up and the system calls directly, with nothing between
//! them and the kernel, and it can be linked beside MLX's own C++ when a
//! caller needs both in one native step.

fn main() {
    println!("cargo:rerun-if-changed=native/expert_record_reader.cpp");
    println!("cargo:rerun-if-changed=native/expert_record_reader.h");
    let mut native_reader_build = cc::Build::new();
    native_reader_build
        .cpp(true)
        .std("c++17")
        .file("native/expert_record_reader.cpp")
        .warnings_into_errors(true);
    // The worker that links this reader supports macOS 14.0. The reader itself
    // is portable POSIX (threads and preadv), so other targets build it as is.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        native_reader_build.flag("-mmacosx-version-min=14.0");
    }
    native_reader_build.compile("astronomical_expert_record_reader");
}
