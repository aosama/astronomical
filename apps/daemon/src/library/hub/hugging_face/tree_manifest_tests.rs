use super::{LfsDocument, TreeEntryDocument, TreeManifestAccumulator};
use crate::library::DownloadPathSelection;
use crate::library::hub::{HuggingFaceHubError, HuggingFaceHubLimits};

const REPOSITORY_ID: &str = "astronomical/ornith-test";
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const GIT_BLOB_SHA1: &str = "89abcdef0123456789abcdef0123456789abcdef";
const LFS_SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn git_file(path: &str, size: u64) -> TreeEntryDocument {
    TreeEntryDocument {
        entry_type: "file".to_owned(),
        size,
        path: path.to_owned(),
        oid: Some(GIT_BLOB_SHA1.to_owned()),
        lfs: None,
        xet_hash: None,
    }
}

fn lfs_file(path: &str, size: u64) -> TreeEntryDocument {
    TreeEntryDocument {
        lfs: Some(LfsDocument {
            oid: LFS_SHA256.to_owned(),
            size,
            pointer_size: 134,
        }),
        ..git_file(path, size)
    }
}

fn directory(path: &str) -> TreeEntryDocument {
    TreeEntryDocument {
        entry_type: "directory".to_owned(),
        size: 0,
        path: path.to_owned(),
        oid: None,
        lfs: None,
        xet_hash: None,
    }
}

fn accept_all(
    tree_manifest: &mut TreeManifestAccumulator,
    tree_entries: Vec<TreeEntryDocument>,
) -> Result<(), HuggingFaceHubError> {
    let complete_repository = DownloadPathSelection::default();
    for tree_entry in tree_entries {
        tree_manifest.accept_entry(tree_entry, &complete_repository)?;
    }
    Ok(())
}

#[test]
fn should_build_a_manifest_of_every_file_with_its_digest_and_total() {
    let mut tree_manifest = TreeManifestAccumulator::new(HuggingFaceHubLimits::default());
    accept_all(
        &mut tree_manifest,
        vec![
            git_file("config.json", 512),
            directory("weights"),
            lfs_file("weights/model.safetensors", 4_096),
        ],
    )
    .expect("a well-formed tree");
    let manifest = tree_manifest
        .finish(REPOSITORY_ID, REVISION)
        .expect("a nonempty manifest");
    assert_eq!(manifest.total_bytes(), 4_608);
    assert_eq!(manifest.files().len(), 2);
    assert_eq!(
        manifest.files()[1].relative_path(),
        "weights/model.safetensors"
    );
}

#[test]
fn should_leave_unselected_files_out_of_the_manifest_but_still_check_them() {
    let weights_only =
        DownloadPathSelection::try_new(Some(vec!["weights/".to_owned()])).expect("a selection");
    let mut tree_manifest = TreeManifestAccumulator::new(HuggingFaceHubLimits::default());
    tree_manifest
        .accept_entry(git_file("README.md", 100), &weights_only)
        .expect("an unselected file");
    tree_manifest
        .accept_entry(lfs_file("weights/model.safetensors", 4_096), &weights_only)
        .expect("a selected file");
    assert!(matches!(
        tree_manifest.accept_entry(git_file("readme.md", 100), &weights_only),
        Err(HuggingFaceHubError::CaseInsensitivePathCollision)
    ));
    let manifest = tree_manifest
        .finish(REPOSITORY_ID, REVISION)
        .expect("a nonempty manifest");
    assert_eq!(manifest.total_bytes(), 4_096);
    assert_eq!(manifest.files().len(), 1);
}

#[test]
fn should_refuse_paths_that_differ_only_in_case() {
    let mut tree_manifest = TreeManifestAccumulator::new(HuggingFaceHubLimits::default());
    assert!(matches!(
        accept_all(
            &mut tree_manifest,
            vec![git_file("Config.json", 1), git_file("config.json", 1)]
        ),
        Err(HuggingFaceHubError::CaseInsensitivePathCollision)
    ));
}

#[test]
fn should_refuse_a_file_that_is_also_a_directory_in_either_order() {
    for tree_entries in [
        vec![
            git_file("weights", 1),
            git_file("weights/model.safetensors", 1),
        ],
        vec![
            git_file("weights/model.safetensors", 1),
            git_file("weights", 1),
        ],
    ] {
        let mut tree_manifest = TreeManifestAccumulator::new(HuggingFaceHubLimits::default());
        assert!(matches!(
            accept_all(&mut tree_manifest, tree_entries),
            Err(HuggingFaceHubError::FilePathHierarchyConflict)
        ));
    }
}

#[test]
fn should_refuse_an_unknown_entry_type() {
    let mut tree_manifest = TreeManifestAccumulator::new(HuggingFaceHubLimits::default());
    let symbolic_link = TreeEntryDocument {
        entry_type: "symlink".to_owned(),
        ..git_file("weights", 0)
    };
    assert!(matches!(
        accept_all(&mut tree_manifest, vec![symbolic_link]),
        Err(HuggingFaceHubError::InvalidEntryType)
    ));
}

#[test]
fn should_accept_exactly_the_entry_limit_and_refuse_one_more() {
    let two_entry_limits = HuggingFaceHubLimits::new(8, 2, 1_024, u64::MAX);
    let mut tree_manifest = TreeManifestAccumulator::new(two_entry_limits);
    accept_all(
        &mut tree_manifest,
        vec![git_file("a.json", 1), git_file("b.json", 1)],
    )
    .expect("two entries fit a two-entry limit");
    assert!(matches!(
        accept_all(&mut tree_manifest, vec![git_file("c.json", 1)]),
        Err(HuggingFaceHubError::TooManyTreeEntries)
    ));
}

#[test]
fn should_accept_exactly_the_byte_limit_and_refuse_one_byte_more() {
    let hundred_byte_limits = HuggingFaceHubLimits::new(8, 16, 1_024, 100);
    let mut tree_manifest = TreeManifestAccumulator::new(hundred_byte_limits);
    accept_all(&mut tree_manifest, vec![git_file("a.bin", 100)])
        .expect("a total equal to the limit fits");
    assert!(matches!(
        accept_all(&mut tree_manifest, vec![git_file("b.bin", 1)]),
        Err(HuggingFaceHubError::TotalBytesTooLarge)
    ));
}

#[test]
fn should_refuse_a_job_whose_durable_metadata_outgrows_its_limit() {
    let small_job_limits =
        HuggingFaceHubLimits::default().with_maximum_job_metadata_bytes(4_096 + 1);
    let mut tree_manifest = TreeManifestAccumulator::new(small_job_limits);
    assert!(matches!(
        accept_all(&mut tree_manifest, vec![git_file("config.json", 1)]),
        Err(HuggingFaceHubError::JobMetadataTooLarge)
    ));
}

#[test]
fn should_refuse_a_tree_without_a_nonempty_selected_file() {
    let mut directories_only = TreeManifestAccumulator::new(HuggingFaceHubLimits::default());
    accept_all(&mut directories_only, vec![directory("weights")]).expect("a directory");
    assert!(matches!(
        directories_only.finish(REPOSITORY_ID, REVISION),
        Err(HuggingFaceHubError::EmptyManifest)
    ));
    let mut empty_file_only = TreeManifestAccumulator::new(HuggingFaceHubLimits::default());
    accept_all(&mut empty_file_only, vec![git_file("empty.txt", 0)]).expect("an empty file");
    assert!(matches!(
        empty_file_only.finish(REPOSITORY_ID, REVISION),
        Err(HuggingFaceHubError::EmptyManifest)
    ));
}

fn refusal_of_single_entry(tree_entry: TreeEntryDocument) -> HuggingFaceHubError {
    let mut tree_manifest = TreeManifestAccumulator::new(HuggingFaceHubLimits::default());
    accept_all(&mut tree_manifest, vec![tree_entry]).expect_err("the entry must be refused")
}

#[test]
fn should_keep_a_lowercase_xet_hash_and_refuse_any_other() {
    let mut tree_manifest = TreeManifestAccumulator::new(HuggingFaceHubLimits::default());
    let xet_file = TreeEntryDocument {
        xet_hash: Some(LFS_SHA256.to_owned()),
        ..lfs_file("model.safetensors", 64)
    };
    accept_all(&mut tree_manifest, vec![xet_file]).expect("a lowercase Xet hash");
    let manifest = tree_manifest
        .finish(REPOSITORY_ID, REVISION)
        .expect("a nonempty manifest");
    assert_eq!(manifest.files()[0].xet_hash(), Some(LFS_SHA256));

    let uppercase_xet_file = TreeEntryDocument {
        xet_hash: Some(LFS_SHA256.to_ascii_uppercase()),
        ..lfs_file("model.safetensors", 64)
    };
    assert!(matches!(
        refusal_of_single_entry(uppercase_xet_file),
        HuggingFaceHubError::InvalidXetHash
    ));
}

#[test]
fn should_refuse_large_file_metadata_that_disagrees_with_the_entry() {
    let mut size_mismatch = lfs_file("model.safetensors", 64);
    if let Some(lfs_document) = size_mismatch.lfs.as_mut() {
        lfs_document.size = 63;
    }
    let mut empty_pointer = lfs_file("model.safetensors", 64);
    if let Some(lfs_document) = empty_pointer.lfs.as_mut() {
        lfs_document.pointer_size = 0;
    }
    for malformed_entry in [size_mismatch, empty_pointer] {
        assert!(matches!(
            refusal_of_single_entry(malformed_entry),
            HuggingFaceHubError::InvalidLfsMetadata
        ));
    }
}

#[test]
fn should_refuse_a_file_with_no_verifiable_digest() {
    let undigested_file = TreeEntryDocument {
        oid: None,
        ..git_file("config.json", 512)
    };
    assert!(matches!(
        refusal_of_single_entry(undigested_file),
        HuggingFaceHubError::MissingIndependentDigest
    ));
}

#[test]
fn should_refuse_a_git_blob_id_that_is_not_lowercase_sha1() {
    let short_blob_id = TreeEntryDocument {
        oid: Some(GIT_BLOB_SHA1[..39].to_owned()),
        ..git_file("config.json", 512)
    };
    assert!(matches!(
        refusal_of_single_entry(short_blob_id),
        HuggingFaceHubError::InvalidGitBlobSha1
    ));
}

#[test]
fn should_refuse_a_directory_that_carries_a_size_or_large_file_metadata() {
    let sized_directory = TreeEntryDocument {
        size: 1,
        ..directory("weights")
    };
    assert!(matches!(
        refusal_of_single_entry(sized_directory),
        HuggingFaceHubError::InvalidEntrySize
    ));
    let directory_with_lfs = TreeEntryDocument {
        lfs: lfs_file("weights", 0).lfs,
        ..directory("weights")
    };
    assert!(matches!(
        refusal_of_single_entry(directory_with_lfs),
        HuggingFaceHubError::InvalidLfsMetadata
    ));
}
