//! Bindgen-generated declarations for the complete official MLX C boundary.
//!
//! Every public `mlx_*` function and type in the pinned headers is declared
//! here, and the generated inventory records exactly which symbols exist so
//! the coverage contract can prove headers and bridge stay in lockstep.

#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unused_imports
)]

include!(concat!(env!("OUT_DIR"), "/mlx_c_bindings.rs"));
include!(concat!(env!("OUT_DIR"), "/bridged_inventory.rs"));

unsafe extern "C" {
    /// Reports an error into MLX's handler machinery.
    ///
    /// Bridged because it is a public MLX-C entry point and this surface
    /// stays complete, not because the bridge calls it: payload trampolines
    /// park their failures thread-locally instead of routing them through
    /// this macro's helper. bindgen 0.73 skips C variadic functions, so the
    /// variadic signature is declared by hand and the build script appends
    /// it to the bridge inventory.
    pub fn mlx_error(
        file: *const ::std::os::raw::c_char,
        line: ::std::os::raw::c_int,
        fmt: *const ::std::os::raw::c_char,
        ...
    ) -> ::std::os::raw::c_int;
}
