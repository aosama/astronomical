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
