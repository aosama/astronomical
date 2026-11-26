use std::fs;
use std::path::Path;

use super::{
    create_temporary_state_file, parse_astro_ssd_prompt_cache_file_hash_from_path,
    remove_temporary_state_file_then_fail, rename_temporary_state_file_into_place,
};
use crate::astro_ssd_prompt_cache::disk_store_error::AstroSsdPromptCacheDiskStoreError;

const LOWERCASE_HASH_HEX: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

#[test]
fn should_parse_a_lowercase_hash_name_with_or_without_an_extension() {
    let expected_hash: [u8; 32] = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
        0xee, 0xff,
    ];
    let block_directory = format!("/astro_ssd_prompt_cache/blocks/{LOWERCASE_HASH_HEX}");
    let state_file = format!("/astro_ssd_prompt_cache/sequence/{LOWERCASE_HASH_HEX}.safetensors");
    assert_eq!(
        parse_astro_ssd_prompt_cache_file_hash_from_path(Path::new(&block_directory)),
        Some(expected_hash)
    );
    assert_eq!(
        parse_astro_ssd_prompt_cache_file_hash_from_path(Path::new(&state_file)),
        Some(expected_hash)
    );
}

#[test]
fn should_refuse_a_name_one_character_short_or_long() {
    let short_name = &LOWERCASE_HASH_HEX[..63];
    let long_name = format!("{LOWERCASE_HASH_HEX}0");
    assert_eq!(
        parse_astro_ssd_prompt_cache_file_hash_from_path(Path::new(short_name)),
        None
    );
    assert_eq!(
        parse_astro_ssd_prompt_cache_file_hash_from_path(Path::new(&long_name)),
        None
    );
}

#[test]
fn should_refuse_a_64_byte_name_with_a_multibyte_character_instead_of_panicking() {
    // One ASCII byte, then a two-byte "é", then 61 more: 64 bytes, but every
    // two-byte hex slice after the first character cuts "é" in half.
    let multibyte_name = format!("a\u{e9}{}", "a".repeat(61));
    assert_eq!(multibyte_name.len(), 64);
    assert_eq!(
        parse_astro_ssd_prompt_cache_file_hash_from_path(Path::new(&multibyte_name)),
        None
    );
}

#[test]
fn should_refuse_names_that_only_number_parsing_accepts() {
    // from_str_radix accepts a leading sign and uppercase digits; neither is
    // a name the astro SSD prompt cache ever writes, and accepting them lets two directories
    // claim one hash on a case-sensitive volume.
    let signed_name = format!("+f{}", &LOWERCASE_HASH_HEX[2..]);
    let uppercase_name = LOWERCASE_HASH_HEX.to_ascii_uppercase();
    for foreign_name in [signed_name, uppercase_name] {
        assert_eq!(
            parse_astro_ssd_prompt_cache_file_hash_from_path(Path::new(&foreign_name)),
            None,
            "{foreign_name}"
        );
    }
}

fn original_failure() -> AstroSsdPromptCacheDiskStoreError {
    AstroSsdPromptCacheDiskStoreError::InvalidStateFileName {
        file_name: "sonnet_18.safetensors".to_owned(),
    }
}

#[test]
fn should_create_a_fresh_temporary_state_file() {
    let staging_directory = tempfile::tempdir().expect("a staging directory");
    let temporary_file_path = staging_directory
        .path()
        .join("sequence_state.safetensors.tmp");

    create_temporary_state_file(&temporary_file_path).expect("a new temporary file");

    assert!(temporary_file_path.is_file());
}

#[test]
fn should_refuse_a_leftover_temporary_state_file() {
    let staging_directory = tempfile::tempdir().expect("a staging directory");
    let temporary_file_path = staging_directory
        .path()
        .join("sequence_state.safetensors.tmp");
    fs::write(&temporary_file_path, b"Shall I compare thee").expect("a leftover file");

    let refusal = create_temporary_state_file(&temporary_file_path)
        .expect_err("a leftover temporary file should be refused");

    assert!(matches!(
        refusal,
        AstroSsdPromptCacheDiskStoreError::OpenTempFile { temp_file_path, .. }
            if temp_file_path == temporary_file_path
    ));
    assert_eq!(
        fs::read(&temporary_file_path).expect("the leftover file"),
        b"Shall I compare thee"
    );
}

#[test]
fn should_refuse_a_symlink_planted_at_the_temporary_state_file_name() {
    let staging_directory = tempfile::tempdir().expect("a staging directory");
    let user_owned_file_path = staging_directory.path().join("user_owned.txt");
    fs::write(&user_owned_file_path, b"Rough winds do shake").expect("a user file");
    let temporary_file_path = staging_directory
        .path()
        .join("sequence_state.safetensors.tmp");
    std::os::unix::fs::symlink(&user_owned_file_path, &temporary_file_path)
        .expect("a planted symlink");

    let refusal = create_temporary_state_file(&temporary_file_path);

    assert!(matches!(
        refusal,
        Err(AstroSsdPromptCacheDiskStoreError::OpenTempFile { .. })
    ));
    assert_eq!(
        fs::read(&user_owned_file_path).expect("the user file"),
        b"Rough winds do shake"
    );
}

#[test]
fn should_remove_the_temporary_file_and_report_the_original_failure() {
    let staging_directory = tempfile::tempdir().expect("a staging directory");
    let temporary_file_path = staging_directory
        .path()
        .join("sequence_state.safetensors.tmp");
    fs::write(&temporary_file_path, b"partial").expect("a temporary file");

    let failure =
        remove_temporary_state_file_then_fail::<()>(&temporary_file_path, original_failure())
            .expect_err("the original failure should be reported");

    assert!(matches!(
        failure,
        AstroSsdPromptCacheDiskStoreError::InvalidStateFileName { .. }
    ));
    assert!(!temporary_file_path.exists());
}

#[test]
fn should_report_the_original_failure_when_the_temporary_file_is_already_gone() {
    let staging_directory = tempfile::tempdir().expect("a staging directory");
    let missing_file_path = staging_directory.path().join("never_created.tmp");

    let failure =
        remove_temporary_state_file_then_fail::<()>(&missing_file_path, original_failure())
            .expect_err("the original failure should be reported");

    assert!(matches!(
        failure,
        AstroSsdPromptCacheDiskStoreError::InvalidStateFileName { .. }
    ));
}

#[test]
fn should_report_the_removal_failure_when_the_temporary_path_cannot_be_removed() {
    let staging_directory = tempfile::tempdir().expect("a staging directory");
    // A directory cannot be removed as a file, and that failure is not
    // "already absent", so it must win over the original failure.
    let unremovable_path = staging_directory.path().join("a_directory.tmp");
    fs::create_dir(&unremovable_path).expect("a directory in the file's place");

    let failure =
        remove_temporary_state_file_then_fail::<()>(&unremovable_path, original_failure())
            .expect_err("a failed removal should be reported");

    assert!(matches!(
        failure,
        AstroSsdPromptCacheDiskStoreError::RemoveAstroSsdPromptCacheFile { .. }
    ));
}

#[test]
fn should_move_the_temporary_file_to_its_final_name() {
    let staging_directory = tempfile::tempdir().expect("a staging directory");
    let temporary_file_path = staging_directory
        .path()
        .join("sequence_state.safetensors.tmp");
    let block_file_path = staging_directory.path().join("sequence_state.safetensors");
    fs::write(&temporary_file_path, b"complete").expect("a temporary file");

    rename_temporary_state_file_into_place(&temporary_file_path, &block_file_path)
        .expect("the rename should succeed");

    assert!(!temporary_file_path.exists());
    assert_eq!(
        fs::read(&block_file_path).expect("the final file"),
        b"complete"
    );
}

#[test]
fn should_replace_a_file_already_at_the_final_name() {
    let staging_directory = tempfile::tempdir().expect("a staging directory");
    let temporary_file_path = staging_directory
        .path()
        .join("sequence_state.safetensors.tmp");
    let block_file_path = staging_directory.path().join("sequence_state.safetensors");
    fs::write(&temporary_file_path, b"new").expect("a temporary file");
    fs::write(&block_file_path, b"old").expect("an older final file");

    rename_temporary_state_file_into_place(&temporary_file_path, &block_file_path)
        .expect("the rename should replace the older file");

    assert_eq!(fs::read(&block_file_path).expect("the final file"), b"new");
}

#[test]
fn should_report_both_paths_when_the_temporary_file_cannot_be_renamed() {
    let staging_directory = tempfile::tempdir().expect("a staging directory");
    let missing_temporary_file_path = staging_directory.path().join("missing.tmp");
    let block_file_path = staging_directory.path().join("sequence_state.safetensors");

    let failure =
        rename_temporary_state_file_into_place(&missing_temporary_file_path, &block_file_path)
            .expect_err("renaming a missing file should fail");

    assert!(matches!(
        failure,
        AstroSsdPromptCacheDiskStoreError::RenameTempFile { temp_file_path, block_file_path: reported_block_file_path, .. }
            if temp_file_path == missing_temporary_file_path
                && reported_block_file_path == block_file_path
    ));
}
