//! User-journey contracts for safe reuse of compiled native products.
//!
//! Fake native payloads keep this suite scripted while exercising the same
//! locking, validation, and publication owner used by the Cargo build script.

use std::{
    error::Error,
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

use super::{NativeBuildProfile, NativeBuildStore};
use crate::native_host_platform::NativeTargetPlatform;

const CORE_NATIVE_IDENTITY: &str =
    "1111111111111111111111111111111111111111111111111111111111111111";
const SIBLING_NATIVE_IDENTITY: &str =
    "2222222222222222222222222222222222222222222222222222222222222222";

#[test]
fn should_reuse_one_complete_native_entry_across_cargo_output_directories() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let build_count = Arc::new(AtomicUsize::new(0));
    let native_build_store = core_native_build_store(store_directory.path(), CORE_NATIVE_IDENTITY);

    let first_artifacts = native_build_store
        .resolve_or_build(fake_builder(Arc::clone(&build_count)))
        .expect("the first caller should publish native products");
    let first_cargo_output_directory = store_directory.path().join("cargo-out-v1");
    let second_cargo_output_directory = store_directory.path().join("cargo-out-v2");
    fs::create_dir_all(&first_cargo_output_directory)
        .expect("the test should create the first Cargo output directory");
    fs::create_dir_all(&second_cargo_output_directory)
        .expect("the test should create the second Cargo output directory");

    let second_artifacts = native_build_store
        .resolve_or_build(fake_builder(Arc::clone(&build_count)))
        .expect("the second caller should reuse native products");

    assert!(first_artifacts.was_built());
    assert!(!second_artifacts.was_built());
    assert_eq!(build_count.load(Ordering::SeqCst), 1);
    assert_eq!(
        first_artifacts.native_library_directory(),
        second_artifacts.native_library_directory()
    );
    assert_eq!(
        first_artifacts.native_library_directory().parent(),
        second_artifacts.native_library_directory().parent()
    );
    assert!(
        first_artifacts
            .native_library_directory()
            .parent()
            .expect("the native library directory has an entry parent")
            .join("include")
            .is_dir()
    );
    assert!(first_artifacts.metallib_path().is_file());
    assert!(first_artifacts.memory_contract_probe_path().is_none());
    assert!(
        !first_artifacts
            .native_library_directory()
            .starts_with(&first_cargo_output_directory)
    );
    assert!(
        !second_artifacts
            .native_library_directory()
            .starts_with(&second_cargo_output_directory)
    );
}

#[test]
fn should_serialize_callers_that_request_the_same_native_identity() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let build_count = Arc::new(AtomicUsize::new(0));
    let first_store = core_native_build_store(store_directory.path(), CORE_NATIVE_IDENTITY);
    let second_store = core_native_build_store(store_directory.path(), CORE_NATIVE_IDENTITY);
    let first_build_count = Arc::clone(&build_count);
    let second_build_count = Arc::clone(&build_count);

    let first_caller = thread::spawn(move || {
        first_store
            .resolve_or_build(move |build_directory| {
                first_build_count.fetch_add(1, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(100));
                create_fake_native_outputs(build_directory, NativeBuildProfile::core())
            })
            .map_err(|build_error| build_error.to_string())
    });
    let second_caller = thread::spawn(move || {
        second_store
            .resolve_or_build(fake_builder(second_build_count))
            .map_err(|build_error| build_error.to_string())
    });

    let first_artifacts = first_caller
        .join()
        .expect("the first native caller should not panic")
        .expect("the first native caller should succeed");
    let second_artifacts = second_caller
        .join()
        .expect("the second native caller should not panic")
        .expect("the second native caller should succeed");

    assert_eq!(build_count.load(Ordering::SeqCst), 1);
    assert_ne!(first_artifacts.was_built(), second_artifacts.was_built());
}

#[test]
fn should_rebuild_only_the_corrupt_native_identity() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let build_count = Arc::new(AtomicUsize::new(0));
    let primary_store = core_native_build_store(store_directory.path(), CORE_NATIVE_IDENTITY);
    let sibling_store = core_native_build_store(store_directory.path(), SIBLING_NATIVE_IDENTITY);
    let primary_artifacts = primary_store
        .resolve_or_build(fake_builder(Arc::clone(&build_count)))
        .expect("the primary entry should publish");
    let sibling_artifacts = sibling_store
        .resolve_or_build(fake_builder(Arc::clone(&build_count)))
        .expect("the sibling entry should publish");
    let sibling_library_bytes = fs::read(sibling_artifacts.mlx_library_path())
        .expect("the test should read the sibling library");
    fs::write(primary_artifacts.mlx_library_path(), b"corrupt")
        .expect("the test should corrupt the primary entry");

    let rebuilt_artifacts = primary_store
        .resolve_or_build(fake_builder(Arc::clone(&build_count)))
        .expect("the corrupt primary entry should rebuild");

    assert!(rebuilt_artifacts.was_built());
    assert_eq!(build_count.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read(sibling_artifacts.mlx_library_path())
            .expect("the sibling entry should remain readable"),
        sibling_library_bytes
    );
}

#[test]
fn should_never_publish_an_interrupted_native_build() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let native_build_store = core_native_build_store(store_directory.path(), CORE_NATIVE_IDENTITY);

    let build_error = native_build_store
        .resolve_or_build(|build_directory| {
            create_fake_native_outputs(build_directory, NativeBuildProfile::core())?;
            Err::<(), Box<dyn Error>>("simulated interrupted native build".into())
        })
        .expect_err("an interrupted build must fail");

    assert!(
        build_error
            .to_string()
            .contains("simulated interrupted native build")
    );
    assert!(!native_build_store.entry_directory().exists());
    let rebuilt_artifacts = native_build_store
        .resolve_or_build(fake_builder(Arc::new(AtomicUsize::new(0))))
        .expect("the next caller should build a complete replacement");
    assert!(rebuilt_artifacts.was_built());
}

#[test]
fn should_publish_and_reuse_the_memory_contract_profile_separately() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let memory_contract_profile = NativeBuildProfile::new(true);
    let native_build_store = NativeBuildStore::new(
        store_directory.path(),
        CORE_NATIVE_IDENTITY,
        NativeTargetPlatform::MacOs,
        memory_contract_profile,
    )
    .expect("the memory-contract store fixture should be valid");

    let first_artifacts = native_build_store
        .resolve_or_build(|build_directory| {
            create_fake_native_outputs(build_directory, memory_contract_profile)
        })
        .expect("the memory-contract profile should publish");
    let second_artifacts = native_build_store
        .resolve_or_build(|_| Err("a valid profile should not rebuild".into()))
        .expect("the memory-contract profile should reuse");

    assert!(first_artifacts.was_built());
    assert!(!second_artifacts.was_built());
    assert!(
        second_artifacts
            .memory_contract_probe_path()
            .expect("the selected profile should expose its probe")
            .is_file()
    );
}

#[test]
fn should_replace_file_and_dangling_symlink_entries_without_following_them() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let build_count = Arc::new(AtomicUsize::new(0));
    let native_build_store = core_native_build_store(store_directory.path(), CORE_NATIVE_IDENTITY);
    native_build_store
        .resolve_or_build(fake_builder(Arc::clone(&build_count)))
        .expect("the first entry should publish");
    fs::remove_dir_all(native_build_store.entry_directory())
        .expect("the test should remove the complete entry");
    fs::write(native_build_store.entry_directory(), b"not-a-directory")
        .expect("the test should replace the entry with a regular file");

    native_build_store
        .resolve_or_build(fake_builder(Arc::clone(&build_count)))
        .expect("a regular-file entry should be replaced");

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;

        fs::remove_dir_all(native_build_store.entry_directory())
            .expect("the test should remove the rebuilt entry");
        symlink(
            store_directory.path().join("missing-target"),
            native_build_store.entry_directory(),
        )
        .expect("the test should create a dangling entry symlink");
        native_build_store
            .resolve_or_build(fake_builder(Arc::clone(&build_count)))
            .expect("a dangling symlink entry should be replaced without following it");
    }

    assert_eq!(build_count.load(Ordering::SeqCst), 3);
    assert!(native_build_store.entry_directory().is_dir());
}

#[cfg(unix)]
#[test]
fn should_reject_symlinked_files_inside_a_native_entry() {
    use std::os::unix::fs::symlink;

    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let build_count = Arc::new(AtomicUsize::new(0));
    let native_build_store = core_native_build_store(store_directory.path(), CORE_NATIVE_IDENTITY);
    let native_artifacts = native_build_store
        .resolve_or_build(fake_builder(Arc::clone(&build_count)))
        .expect("the first entry should publish");
    let external_library_path = store_directory.path().join("external-libmlx.a");
    fs::write(&external_library_path, b"external")
        .expect("the test should create an external native library");
    fs::remove_file(native_artifacts.mlx_library_path())
        .expect("the test should remove the published native library");
    symlink(&external_library_path, native_artifacts.mlx_library_path())
        .expect("the test should create a malicious symlink");

    let rebuilt_artifacts = native_build_store
        .resolve_or_build(fake_builder(Arc::clone(&build_count)))
        .expect("a symlinked entry should be discarded and rebuilt");

    assert!(rebuilt_artifacts.was_built());
    assert_eq!(build_count.load(Ordering::SeqCst), 2);
    assert!(
        !fs::symlink_metadata(rebuilt_artifacts.mlx_library_path())
            .expect("the rebuilt library should have metadata")
            .file_type()
            .is_symlink()
    );
}

#[test]
fn should_round_trip_every_profile_through_its_identity_name() {
    for native_build_profile in [NativeBuildProfile::core(), NativeBuildProfile::new(true)] {
        let identity_name = native_build_profile.identity_name();
        let restored_profile = NativeBuildProfile::from_identity_name(identity_name)
            .unwrap_or_else(|| panic!("profile {identity_name} should round-trip"));

        assert_eq!(restored_profile.identity_name(), identity_name);
    }
}

#[test]
fn should_reject_an_unknown_identity_name() {
    assert!(NativeBuildProfile::from_identity_name("core+unknown-probe").is_none());
}

fn core_native_build_store(store_root: &Path, native_identity: &str) -> NativeBuildStore {
    NativeBuildStore::new(
        store_root,
        native_identity,
        NativeTargetPlatform::MacOs,
        NativeBuildProfile::core(),
    )
    .expect("the native store fixture should be valid")
}

fn fake_builder(build_count: Arc<AtomicUsize>) -> impl FnOnce(&Path) -> Result<(), Box<dyn Error>> {
    move |build_directory| {
        build_count.fetch_add(1, Ordering::SeqCst);
        create_fake_native_outputs(build_directory, NativeBuildProfile::core())
    }
}

fn create_fake_native_outputs(
    build_directory: &Path,
    native_build_profile: NativeBuildProfile,
) -> Result<(), Box<dyn Error>> {
    let mlx_c_header_directory = build_directory.join("_deps/mlx_c-src/mlx/c");
    let library_directory = build_directory.join("lib");
    let metallib_directory = build_directory.join("_deps/mlx-build/mlx/backend/metal/kernels");
    fs::create_dir_all(&mlx_c_header_directory)?;
    fs::create_dir_all(&library_directory)?;
    fs::create_dir_all(&metallib_directory)?;
    fs::write(mlx_c_header_directory.join("mlx.h"), b"mlx-c-header")?;
    fs::write(library_directory.join("libmlx.a"), b"mlx-library")?;
    fs::write(library_directory.join("libmlxc.a"), b"mlxc-library")?;
    fs::write(metallib_directory.join("mlx.metallib"), b"mlx-metallib")?;

    if native_build_profile.should_build_memory_contract_probe() {
        let binary_directory = build_directory.join("bin");
        fs::create_dir_all(&binary_directory)?;
        fs::write(binary_directory.join("mlx_memory_contract_probe"), b"probe")?;
    }
    Ok(())
}

fn linux_native_build_store(store_root: &Path) -> NativeBuildStore {
    NativeBuildStore::new(
        store_root,
        CORE_NATIVE_IDENTITY,
        NativeTargetPlatform::Linux,
        NativeBuildProfile::core(),
    )
    .expect("the Linux native store fixture should be valid")
}

/// A Linux CMake tree: the shared archives and header, MLX's installed CMake
/// package, and no Metallib.
fn create_fake_linux_native_outputs(build_directory: &Path) -> Result<(), Box<dyn Error>> {
    let mlx_c_header_directory = build_directory.join("_deps/mlx_c-src/mlx/c");
    let library_directory = build_directory.join("lib");
    fs::create_dir_all(&mlx_c_header_directory)?;
    fs::create_dir_all(&library_directory)?;
    fs::write(mlx_c_header_directory.join("mlx.h"), b"mlx-c-header")?;
    fs::write(library_directory.join("libmlx.a"), b"mlx-library")?;
    fs::write(library_directory.join("libmlxc.a"), b"mlxc-library")?;
    let mlx_package_directory = build_directory.join("mlx-package/share/cmake/MLX");
    fs::create_dir_all(&mlx_package_directory)?;
    fs::write(
        mlx_package_directory.join("MLXTargets.cmake"),
        b"set_target_properties(mlx PROPERTIES\n  INTERFACE_LINK_LIBRARIES \"dl\"\n)\n",
    )?;
    Ok(())
}

#[test]
fn should_publish_and_reuse_a_linux_entry_without_a_metallib() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let native_build_store = linux_native_build_store(store_directory.path());

    let first_artifacts = native_build_store
        .resolve_or_build(create_fake_linux_native_outputs)
        .expect("a Linux entry should publish without a Metallib");
    let second_artifacts = native_build_store
        .resolve_or_build(|_| Err("a valid Linux entry should not rebuild".into()))
        .expect("a Linux entry should be reused");

    assert!(first_artifacts.was_built());
    assert!(!second_artifacts.was_built());
    assert!(second_artifacts.mlx_targets_path().is_file());
    assert!(!second_artifacts.metallib_path().exists());
}

#[test]
fn should_refuse_to_publish_a_linux_entry_without_mlx_targets() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let native_build_store = linux_native_build_store(store_directory.path());

    let publication_error = native_build_store
        .resolve_or_build(|build_directory| {
            create_fake_linux_native_outputs(build_directory)?;
            fs::remove_file(build_directory.join("mlx-package/share/cmake/MLX/MLXTargets.cmake"))?;
            Ok(())
        })
        .expect_err("a Linux entry needs the libraries MLX links");

    assert!(
        publication_error.to_string().contains("MLXTargets.cmake"),
        "the error should name the missing list: {publication_error}"
    );
    assert!(!native_build_store.entry_directory().exists());
}

#[test]
fn should_refuse_to_publish_a_macos_entry_without_its_metallib() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let native_build_store = core_native_build_store(store_directory.path(), CORE_NATIVE_IDENTITY);

    let publication_result = native_build_store.resolve_or_build(create_fake_linux_native_outputs);

    assert!(
        publication_result.is_err(),
        "a macOS entry must carry the Metallib"
    );
    assert!(!native_build_store.entry_directory().exists());
}

#[test]
fn should_build_every_identity_in_the_same_reusable_folder() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let builder_folders = Arc::new(std::sync::Mutex::new(Vec::new()));

    for native_identity in [CORE_NATIVE_IDENTITY, SIBLING_NATIVE_IDENTITY] {
        let recorded_folders = Arc::clone(&builder_folders);
        let build_count = Arc::new(AtomicUsize::new(0));
        let recording_builder = fake_builder(build_count);
        core_native_build_store(store_directory.path(), native_identity)
            .resolve_or_build(move |cmake_build_directory: &Path| {
                recorded_folders
                    .lock()
                    .unwrap()
                    .push(cmake_build_directory.to_path_buf());
                recording_builder(cmake_build_directory)
            })
            .expect("each identity should publish");
    }

    let builder_folders = builder_folders.lock().unwrap();
    assert_eq!(builder_folders.len(), 2);
    assert_eq!(
        builder_folders[0], builder_folders[1],
        "a new identity must compile at the same paths so sccache can serve unchanged objects"
    );
}

fn linux_cuda_native_build_store(store_root: &Path) -> NativeBuildStore {
    NativeBuildStore::new(
        store_root,
        CORE_NATIVE_IDENTITY,
        NativeTargetPlatform::LinuxCuda,
        NativeBuildProfile::core(),
    )
    .expect("the Linux CUDA native store fixture should be valid")
}

/// A Linux CUDA CMake tree: the Linux products, the two CUDA fact files, and
/// MLX's installed run-time compiler headers, one nested header each.
fn create_fake_linux_cuda_native_outputs(build_directory: &Path) -> Result<(), Box<dyn Error>> {
    create_fake_linux_native_outputs(build_directory)?;
    let cuda_facts_directory = build_directory.join("cuda");
    fs::create_dir_all(&cuda_facts_directory)?;
    fs::write(cuda_facts_directory.join("link-libraries.txt"), b"-lcuda\n")?;
    fs::write(
        cuda_facts_directory.join("toolkit-include-directory.txt"),
        b"/toolkit/include\n",
    )?;
    for (header_family, nested_header) in [
        ("cute", "numeric/numeric_types.hpp"),
        ("cutlass", "numeric_types.h"),
        ("cccl", "cuda/std/type_traits"),
    ] {
        let header_path = build_directory
            .join("mlx-package/include")
            .join(header_family)
            .join(nested_header);
        fs::create_dir_all(header_path.parent().expect("a nested header has a folder"))?;
        fs::write(&header_path, header_family.as_bytes())?;
    }
    Ok(())
}

#[test]
fn should_publish_and_reuse_a_linux_cuda_entry_with_its_run_time_compiler_headers() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let native_build_store = linux_cuda_native_build_store(store_directory.path());

    native_build_store
        .resolve_or_build(create_fake_linux_cuda_native_outputs)
        .expect("a Linux CUDA entry should publish");
    let reused_artifacts = native_build_store
        .resolve_or_build(|_| Err("a valid Linux CUDA entry should not rebuild".into()))
        .expect("a Linux CUDA entry should be reused");

    let jit_headers_directory = reused_artifacts.cuda_jit_headers_directory();
    assert!(
        jit_headers_directory
            .join("cute/numeric/numeric_types.hpp")
            .is_file()
    );
    assert!(
        jit_headers_directory
            .join("cutlass/numeric_types.h")
            .is_file()
    );
    assert!(
        jit_headers_directory
            .join("cccl/cuda/std/type_traits")
            .is_file()
    );
    assert!(reused_artifacts.cuda_link_libraries_path().is_file());
    assert!(
        reused_artifacts
            .cuda_toolkit_include_directory_path()
            .is_file()
    );
}

#[test]
fn should_publish_an_empty_header_family_folder_as_a_folder() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let native_build_store = linux_cuda_native_build_store(store_directory.path());

    let published_artifacts = native_build_store
        .resolve_or_build(|build_directory| {
            create_fake_linux_cuda_native_outputs(build_directory)?;
            let cccl_directory = build_directory.join("mlx-package/include/cccl");
            fs::remove_dir_all(&cccl_directory)?;
            fs::create_dir_all(&cccl_directory)?;
            Ok(())
        })
        .expect("an empty header folder is still a folder");

    assert!(
        published_artifacts
            .cuda_jit_headers_directory()
            .join("cccl")
            .is_dir()
    );
}

#[test]
fn should_refuse_to_publish_a_linux_cuda_entry_without_its_cutlass_headers() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let native_build_store = linux_cuda_native_build_store(store_directory.path());

    let publication_error = native_build_store
        .resolve_or_build(|build_directory| {
            create_fake_linux_cuda_native_outputs(build_directory)?;
            fs::remove_dir_all(build_directory.join("mlx-package/include/cutlass"))?;
            Ok(())
        })
        .expect_err("a Linux CUDA entry needs every run-time compiler header folder");

    assert!(
        publication_error.to_string().contains("cutlass"),
        "the error should name the missing folder: {publication_error}"
    );
    assert!(!native_build_store.entry_directory().exists());
}

#[cfg(unix)]
#[test]
fn should_refuse_to_publish_run_time_compiler_headers_that_hold_a_symlink() {
    let store_directory = tempfile::tempdir().expect("the test should create a temporary store");
    let native_build_store = linux_cuda_native_build_store(store_directory.path());

    let publication_error = native_build_store
        .resolve_or_build(|build_directory| {
            create_fake_linux_cuda_native_outputs(build_directory)?;
            std::os::unix::fs::symlink(
                "/etc/hostname",
                build_directory.join("mlx-package/include/cute/linked_header.hpp"),
            )?;
            Ok(())
        })
        .expect_err("a symlink inside the headers must never be published");

    assert!(
        publication_error.to_string().contains("symlink"),
        "the error should name the symlink: {publication_error}"
    );
    assert!(!native_build_store.entry_directory().exists());
}
