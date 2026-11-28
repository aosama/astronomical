use std::fs;
use std::path::PathBuf;

use crate::linux_gpu_backend::{
    cuda_architectures_from_compute_capabilities, cuda_compiler_major_version_from_version_output,
    cudnn_version_from_version_header, find_executable_in,
};

/// The version lines of cuDNN 9's `cudnn_version.h`.
const CUDNN_9_VERSION_HEADER: &str = "#ifndef CUDNN_VERSION_H_\n\
#define CUDNN_MAJOR 9\n\
#define CUDNN_MINOR 27\n\
#define CUDNN_PATCHLEVEL 0\n\
#define CUDNN_VERSION (CUDNN_MAJOR * 10000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)\n";

/// `nvcc --version` as CUDA 13 prints it.
const CUDA_13_COMPILER_VERSION_OUTPUT: &str = "nvcc: NVIDIA (R) Cuda compiler driver\n\
Copyright (c) 2005-2026 NVIDIA Corporation\n\
Cuda compilation tools, release 13.0, V13.0.88\n\
Build cuda_13.0.r13.0/compiler.36424714_0\n";

// --- positive ---

#[test]
fn should_name_an_ampere_gpu_by_its_architecture_number() {
    assert_eq!(
        cuda_architectures_from_compute_capabilities("8.6\n").as_deref(),
        Some("86")
    );
}

#[test]
fn should_name_a_hopper_or_later_gpu_with_mlxs_architecture_specific_suffix() {
    assert_eq!(
        cuda_architectures_from_compute_capabilities("9.0\n").as_deref(),
        Some("90a")
    );
    assert_eq!(
        cuda_architectures_from_compute_capabilities("12.0\n").as_deref(),
        Some("120a")
    );
}

#[test]
fn should_find_an_executable_in_the_first_directory_that_holds_it() {
    let search_root =
        std::env::temp_dir().join(format!("linux_gpu_backend_tests_{}", std::process::id()));
    let empty_directory = search_root.join("empty");
    let first_holding_directory = search_root.join("first");
    let second_holding_directory = search_root.join("second");
    for search_directory in [
        &empty_directory,
        &first_holding_directory,
        &second_holding_directory,
    ] {
        fs::create_dir_all(search_directory).unwrap();
    }
    fs::write(first_holding_directory.join("nvcc"), b"").unwrap();
    fs::write(second_holding_directory.join("nvcc"), b"").unwrap();
    let found_path = find_executable_in(
        [
            empty_directory,
            first_holding_directory.clone(),
            second_holding_directory,
        ],
        "nvcc",
    );
    assert_eq!(found_path, Some(first_holding_directory.join("nvcc")));
    fs::remove_dir_all(&search_root).unwrap();
}

#[test]
fn should_read_the_major_version_from_the_cuda_compilers_release_line() {
    assert_eq!(
        cuda_compiler_major_version_from_version_output(CUDA_13_COMPILER_VERSION_OUTPUT),
        Some(13)
    );
}

#[test]
fn should_read_the_cudnn_version_from_its_version_header() {
    assert_eq!(
        cudnn_version_from_version_header(CUDNN_9_VERSION_HEADER).as_deref(),
        Some("9.27.0")
    );
}

// --- edge / boundary ---

#[test]
fn should_not_take_a_longer_macro_name_for_the_cudnn_major_version() {
    let version_header = "#define CUDNN_MAJOR_SUFFIX 1\n#define CUDNN_MAJOR 9\n\
#define CUDNN_MINOR 0\n#define CUDNN_PATCHLEVEL 3\n";
    assert_eq!(
        cudnn_version_from_version_header(version_header).as_deref(),
        Some("9.0.3")
    );
}

#[test]
fn should_list_each_architecture_once_in_the_order_first_seen() {
    assert_eq!(
        cuda_architectures_from_compute_capabilities("8.6\n9.0\n8.6\n").as_deref(),
        Some("86;90a")
    );
}

#[test]
fn should_skip_blank_and_padded_lines() {
    assert_eq!(
        cuda_architectures_from_compute_capabilities("\n  7.5  \n\n").as_deref(),
        Some("75")
    );
}

#[test]
fn should_name_a_two_digit_major_compute_capability_with_the_suffix() {
    assert_eq!(
        cuda_architectures_from_compute_capabilities("10.0\n").as_deref(),
        Some("100a")
    );
}

#[test]
fn should_accept_the_oldest_supported_gpu_and_drop_the_one_just_below_it() {
    assert_eq!(
        cuda_architectures_from_compute_capabilities("7.5\n").as_deref(),
        Some("75")
    );
    assert_eq!(cuda_architectures_from_compute_capabilities("7.2\n"), None);
}

#[test]
fn should_keep_only_the_supported_gpus_of_a_mixed_host() {
    assert_eq!(
        cuda_architectures_from_compute_capabilities("6.1\n8.6\n[N/A]\n").as_deref(),
        Some("86")
    );
}

// --- negative ---

#[test]
fn should_report_no_architecture_for_a_host_with_only_older_gpus() {
    assert_eq!(
        cuda_architectures_from_compute_capabilities("6.1\n7.0\n"),
        None
    );
}

#[test]
fn should_reject_malformed_compute_capabilities() {
    assert_eq!(cuda_architectures_from_compute_capabilities("8\n"), None);
    assert_eq!(cuda_architectures_from_compute_capabilities("8.x\n"), None);
    assert_eq!(cuda_architectures_from_compute_capabilities("x.6\n"), None);
}

#[test]
fn should_report_no_cuda_compiler_version_without_a_release_line() {
    assert_eq!(cuda_compiler_major_version_from_version_output(""), None);
    assert_eq!(
        cuda_compiler_major_version_from_version_output("nvcc: NVIDIA (R) Cuda compiler driver\n"),
        None
    );
    assert_eq!(
        cuda_compiler_major_version_from_version_output(
            "Cuda compilation tools, release thirteen\n"
        ),
        None
    );
}

#[test]
fn should_read_an_older_cuda_compiler_as_its_own_major_version() {
    assert_eq!(
        cuda_compiler_major_version_from_version_output(
            "Cuda compilation tools, release 11.8, V11.8.89\n"
        ),
        Some(11)
    );
}

#[test]
fn should_report_no_architecture_when_the_driver_reports_none() {
    assert_eq!(cuda_architectures_from_compute_capabilities(""), None);
    assert_eq!(
        cuda_architectures_from_compute_capabilities("[N/A]\nNo devices were found\n"),
        None
    );
}

#[test]
fn should_find_nothing_where_no_directory_holds_the_executable() {
    let empty_directory: PathBuf =
        std::env::temp_dir().join(format!("linux_gpu_backend_empty_{}", std::process::id()));
    fs::create_dir_all(&empty_directory).unwrap();
    assert_eq!(find_executable_in([empty_directory.clone()], "nvcc"), None);
    fs::remove_dir_all(&empty_directory).unwrap();
}

#[test]
fn should_report_no_cudnn_version_when_a_definition_is_missing_or_not_a_number() {
    assert_eq!(cudnn_version_from_version_header(""), None);
    assert_eq!(
        cudnn_version_from_version_header("#define CUDNN_MAJOR 9\n#define CUDNN_MINOR 27\n"),
        None
    );
    assert_eq!(
        cudnn_version_from_version_header(
            "#define CUDNN_MAJOR nine\n#define CUDNN_MINOR 27\n#define CUDNN_PATCHLEVEL 0\n"
        ),
        None
    );
}
