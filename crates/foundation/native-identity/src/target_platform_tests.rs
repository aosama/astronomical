use crate::target_platform::{NativeTargetPlatform, UnsupportedTargetPlatform};

#[test]
fn should_read_macos_from_apple_darwin_triples() {
    for target_triple in ["aarch64-apple-darwin", "x86_64-apple-darwin"] {
        assert_eq!(
            NativeTargetPlatform::of_target_triple(target_triple),
            Ok(NativeTargetPlatform::MacOs),
            "{target_triple} is a macOS target"
        );
    }
}

#[test]
fn should_read_linux_from_every_linux_c_library_and_architecture() {
    for target_triple in [
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "x86_64-unknown-linux-musl",
    ] {
        assert_eq!(
            NativeTargetPlatform::of_target_triple(target_triple),
            Ok(NativeTargetPlatform::Linux),
            "{target_triple} is a Linux target"
        );
    }
}

#[test]
fn should_reject_operating_systems_without_a_native_recipe() {
    for target_triple in [
        "x86_64-pc-windows-msvc",
        "aarch64-apple-ios",
        "aarch64-linux-android",
        "wasm32-unknown-unknown",
        "",
    ] {
        assert_eq!(
            NativeTargetPlatform::of_target_triple(target_triple),
            Err(UnsupportedTargetPlatform {
                target_triple: target_triple.to_owned()
            }),
            "{target_triple:?} has no native recipe"
        );
    }
}

#[test]
fn should_not_mistake_a_triple_that_only_mentions_darwin_for_macos() {
    assert_eq!(
        NativeTargetPlatform::of_target_triple("x86_64-unknown-darwin-gnu"),
        Err(UnsupportedTargetPlatform {
            target_triple: "x86_64-unknown-darwin-gnu".to_owned()
        })
    );
}

// --- the Linux GPU backend (issue #2699) ---

#[test]
fn should_promote_a_linux_target_to_cuda_when_the_host_can_build_it() {
    assert_eq!(
        NativeTargetPlatform::Linux.with_gpu_backend(true),
        NativeTargetPlatform::LinuxCuda
    );
}

#[test]
fn should_keep_the_cpu_recipe_on_a_linux_host_without_cuda() {
    assert_eq!(
        NativeTargetPlatform::Linux.with_gpu_backend(false),
        NativeTargetPlatform::Linux
    );
}

#[test]
fn should_never_promote_macos_and_treat_both_linux_recipes_as_linux() {
    assert_eq!(
        NativeTargetPlatform::MacOs.with_gpu_backend(true),
        NativeTargetPlatform::MacOs
    );
    assert!(NativeTargetPlatform::Linux.is_linux());
    assert!(NativeTargetPlatform::LinuxCuda.is_linux());
    assert!(!NativeTargetPlatform::MacOs.is_linux());
}
