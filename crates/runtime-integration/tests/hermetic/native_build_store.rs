//! User-journey contracts for safe reuse of compiled native products.
//!
//! Fake native payloads keep this suite hermetic while exercising the same
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

#[path = "../../build_native_store.rs"]
mod build_native_store;

use build_native_store::{NativeBuildProfile, NativeBuildStore};

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
    assert!(first_artifacts.include_directory().is_dir());
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
    let memory_contract_profile = NativeBuildProfile::new(true, false);
    let native_build_store = NativeBuildStore::new(
        store_directory.path(),
        CORE_NATIVE_IDENTITY,
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

fn core_native_build_store(store_root: &Path, native_identity: &str) -> NativeBuildStore {
    NativeBuildStore::new(
        store_root,
        native_identity,
        NativeBuildProfile::new(false, false),
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
    if native_build_profile.should_build_experimental_aligned_expert_packs() {
        fs::write(
            library_directory.join("libastronomical_metal_expert_loader.a"),
            b"experimental-library",
        )?;
    }
    Ok(())
}
