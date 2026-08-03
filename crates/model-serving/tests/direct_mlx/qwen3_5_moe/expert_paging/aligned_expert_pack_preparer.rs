use std::{
    fs::{self, File},
    os::unix::fs::FileExt,
};

use astronomical_model_serving::{
    AlignedExpertPackBuildRequest, AlignedExpertPackPreparationError, AlignedExpertPackPreparer,
    build_aligned_expert_pack, read_aligned_expert_pack_header,
    validate_aligned_expert_pack_payload,
};

use super::aligned_expert_pack::write_synthetic_expert_source_for_layer;

#[test]
fn should_publish_a_complete_aligned_expert_pack_set_after_every_layer_validates() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let model_directory = temporary_directory.path().join("synthetic-model");
    fs::create_dir(&model_directory).expect("the synthetic model directory should be creatable");
    let first_source_path = model_directory.join("first-source.bin");
    let second_source_path = model_directory.join("second-source.bin");
    let layer_plans = vec![
        write_synthetic_expert_source_for_layer(&first_source_path, 0),
        write_synthetic_expert_source_for_layer(&second_source_path, 1),
    ];
    let preparer = AlignedExpertPackPreparer::from_layer_plans(
        &model_directory,
        "synthetic-model",
        "revision-1",
        layer_plans,
    )
    .expect("the synthetic aligned expert-pack preparation should plan");
    let mut progress_events = Vec::new();

    let preparation_report = preparer
        .prepare(false, |progress_event| progress_events.push(progress_event))
        .expect("the complete aligned expert-pack set should prepare");

    assert_eq!(preparation_report.completed_layer_count, 2);
    assert!(!preparation_report.reused_existing_pack_set);
    assert_eq!(progress_events.len(), 2);
    assert!(preparation_report.final_revision_directory.is_dir());
    assert!(
        preparation_report
            .final_revision_directory
            .join("layer-0.aligned-expert-pack")
            .is_file()
    );
    assert!(
        preparation_report
            .final_revision_directory
            .join("layer-1.aligned-expert-pack")
            .is_file()
    );
    assert!(
        !model_directory
            .join(".astronomical-aligned-expert-packs")
            .join(".revision-1.preparing")
            .exists(),
        "a complete preparation should publish by renaming the staging directory"
    );
}

#[test]
fn should_resume_from_a_valid_layer_left_in_the_staging_revision() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let model_directory = temporary_directory.path().join("synthetic-model");
    fs::create_dir(&model_directory).expect("the synthetic model directory should be creatable");
    let first_layer_plan =
        write_synthetic_expert_source_for_layer(&model_directory.join("first-source.bin"), 0);
    let second_layer_plan =
        write_synthetic_expert_source_for_layer(&model_directory.join("second-source.bin"), 1);
    let staging_revision_directory = model_directory
        .join(".astronomical-aligned-expert-packs")
        .join(".revision-1.preparing");
    fs::create_dir_all(&staging_revision_directory)
        .expect("the interrupted staging revision should be creatable");
    let staged_first_layer_path = staging_revision_directory.join("layer-0.aligned-expert-pack");
    let staged_first_layer_header = build_aligned_expert_pack(
        &staged_first_layer_path,
        &AlignedExpertPackBuildRequest {
            model_id: "synthetic-model",
            model_revision: "revision-1",
            layer_index: 0,
            layer_plan: &first_layer_plan,
        },
    )
    .expect("the simulated interrupted run should leave one valid staged layer");
    let preparer = AlignedExpertPackPreparer::from_layer_plans(
        &model_directory,
        "synthetic-model",
        "revision-1",
        vec![first_layer_plan, second_layer_plan],
    )
    .expect("the resumable preparation should plan");

    let preparation_inspection = preparer
        .inspect()
        .expect("inspection should recognize the valid staged layer");
    assert_eq!(
        preparation_inspection.remaining_pack_byte_count,
        preparation_inspection.total_pack_byte_count
            - staged_first_layer_header.expected_pack_byte_count,
    );

    let preparation_report = preparer
        .prepare(false, |_progress_event| {})
        .expect("preparation should resume and publish both layers");
    assert_eq!(preparation_report.completed_layer_count, 2);
    assert!(
        preparation_report
            .final_revision_directory
            .join("layer-0.aligned-expert-pack")
            .is_file()
    );
    assert!(
        preparation_report
            .final_revision_directory
            .join("layer-1.aligned-expert-pack")
            .is_file()
    );
}

#[test]
fn should_rebuild_a_staged_layer_when_its_payload_no_longer_matches_the_source() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let model_directory = temporary_directory.path().join("synthetic-model");
    fs::create_dir(&model_directory).expect("the synthetic model directory should be creatable");
    let first_layer_plan =
        write_synthetic_expert_source_for_layer(&model_directory.join("first-source.bin"), 0);
    let second_layer_plan =
        write_synthetic_expert_source_for_layer(&model_directory.join("second-source.bin"), 1);
    let staging_revision_directory = model_directory
        .join(".astronomical-aligned-expert-packs")
        .join(".revision-1.preparing");
    fs::create_dir_all(&staging_revision_directory)
        .expect("the interrupted staging revision should be creatable");
    let staged_first_layer_path = staging_revision_directory.join("layer-0.aligned-expert-pack");
    let staged_first_layer_header = build_aligned_expert_pack(
        &staged_first_layer_path,
        &AlignedExpertPackBuildRequest {
            model_id: "synthetic-model",
            model_revision: "revision-1",
            layer_index: 0,
            layer_plan: &first_layer_plan,
        },
    )
    .expect("the simulated interrupted run should leave one staged layer");
    let first_payload_offset =
        staged_first_layer_header.tensor_descriptors[0].pack_segment_offset_bytes;
    let staged_pack_file = File::options()
        .read(true)
        .write(true)
        .open(&staged_first_layer_path)
        .expect("the staged pack should reopen for corruption");
    let mut original_payload_byte = [0_u8; 1];
    staged_pack_file
        .read_at(&mut original_payload_byte, first_payload_offset)
        .expect("the staged payload byte should be readable");
    staged_pack_file
        .write_at(&[original_payload_byte[0] ^ 0xFF], first_payload_offset)
        .expect("the staged payload byte should be corruptible");
    let preparer = AlignedExpertPackPreparer::from_layer_plans(
        &model_directory,
        "synthetic-model",
        "revision-1",
        vec![first_layer_plan.clone(), second_layer_plan],
    )
    .expect("the resumable preparation should plan");

    let preparation_inspection = preparer
        .inspect()
        .expect("inspection should reject the corrupted staged payload");
    assert_eq!(
        preparation_inspection.remaining_pack_byte_count,
        preparation_inspection.total_pack_byte_count,
        "a corrupted staged layer must remain part of the required output bytes"
    );

    let preparation_report = preparer
        .prepare(false, |_progress_event| {})
        .expect("preparation should rebuild the corrupted staged layer");
    let published_first_layer_path = preparation_report
        .final_revision_directory
        .join("layer-0.aligned-expert-pack");
    let published_first_layer_header = read_aligned_expert_pack_header(&published_first_layer_path)
        .expect("the rebuilt layer header should be readable");
    validate_aligned_expert_pack_payload(
        &published_first_layer_path,
        &published_first_layer_header,
        &first_layer_plan,
    )
    .expect("the rebuilt staged layer should preserve source payload parity");
}

#[test]
fn should_reuse_a_complete_valid_revision_without_rewriting_layers() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let model_directory = temporary_directory.path().join("synthetic-model");
    fs::create_dir(&model_directory).expect("the synthetic model directory should be creatable");
    let source_path = model_directory.join("source.bin");
    let preparer = AlignedExpertPackPreparer::from_layer_plans(
        &model_directory,
        "synthetic-model",
        "revision-1",
        vec![write_synthetic_expert_source_for_layer(&source_path, 0)],
    )
    .expect("the synthetic preparation should plan");
    let first_report = preparer
        .prepare(false, |_progress_event| {})
        .expect("the first preparation should publish");
    let pack_path = first_report
        .final_revision_directory
        .join("layer-0.aligned-expert-pack");
    let pack_modified_before_reuse = fs::metadata(&pack_path)
        .expect("the published pack should have metadata")
        .modified()
        .expect("the published pack should have a modification time");
    let mut repeated_progress_event_count = 0;

    let repeated_report = preparer
        .prepare(false, |_progress_event| repeated_progress_event_count += 1)
        .expect("the complete valid revision should be reusable");

    assert!(repeated_report.reused_existing_pack_set);
    assert_eq!(repeated_progress_event_count, 0);
    assert_eq!(
        fs::metadata(pack_path)
            .expect("the reused pack should retain metadata")
            .modified()
            .expect("the reused pack should retain a modification time"),
        pack_modified_before_reuse,
    );
}

#[test]
fn should_reject_a_complete_revision_whose_payload_no_longer_matches_the_source() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let model_directory = temporary_directory.path().join("synthetic-model");
    fs::create_dir(&model_directory).expect("the synthetic model directory should be creatable");
    let source_path = model_directory.join("source.bin");
    let layer_plan = write_synthetic_expert_source_for_layer(&source_path, 0);
    let preparer = AlignedExpertPackPreparer::from_layer_plans(
        &model_directory,
        "synthetic-model",
        "revision-1",
        vec![layer_plan],
    )
    .expect("the synthetic preparation should plan");
    let initial_report = preparer
        .prepare(false, |_progress_event| {})
        .expect("the initial preparation should publish");
    let published_pack_path = initial_report
        .final_revision_directory
        .join("layer-0.aligned-expert-pack");
    let published_pack_header = read_aligned_expert_pack_header(&published_pack_path)
        .expect("the published pack header should be readable");
    let first_payload_offset =
        published_pack_header.tensor_descriptors[0].pack_segment_offset_bytes;
    let published_pack_file = File::options()
        .read(true)
        .write(true)
        .open(&published_pack_path)
        .expect("the published pack should reopen for corruption");
    let mut original_payload_byte = [0_u8; 1];
    published_pack_file
        .read_at(&mut original_payload_byte, first_payload_offset)
        .expect("the published payload byte should be readable");
    published_pack_file
        .write_at(&[original_payload_byte[0] ^ 0xFF], first_payload_offset)
        .expect("the published payload byte should be corruptible");

    let repeated_preparation_outcome = preparer.prepare(false, |_progress_event| {});

    assert!(matches!(
        repeated_preparation_outcome,
        Err(AlignedExpertPackPreparationError::InvalidExistingRevision { .. })
    ));
}

#[test]
fn should_reject_a_concurrent_preparer_for_the_same_model_revision() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let model_directory = temporary_directory.path().join("synthetic-model");
    fs::create_dir(&model_directory).expect("the synthetic model directory should be creatable");
    let source_path = model_directory.join("source.bin");
    let preparer = AlignedExpertPackPreparer::from_layer_plans(
        &model_directory,
        "synthetic-model",
        "revision-1",
        vec![write_synthetic_expert_source_for_layer(&source_path, 0)],
    )
    .expect("the synthetic preparation should plan");
    let pack_root_directory = model_directory.join(".astronomical-aligned-expert-packs");
    fs::create_dir(&pack_root_directory).expect("the generated pack root should be creatable");
    let lock_path = pack_root_directory.join(".revision-1.lock");
    let lock_file = File::options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .expect("the test should open the preparation lock");
    lock_file
        .try_lock()
        .expect("the first simulated preparer should own the lock");

    let preparation_outcome = preparer.prepare(false, |_progress_event| {});

    assert!(matches!(
        preparation_outcome,
        Err(AlignedExpertPackPreparationError::PreparationAlreadyRunning { .. })
    ));
}

#[test]
fn should_require_replace_before_rebuilding_an_invalid_final_revision() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let model_directory = temporary_directory.path().join("synthetic-model");
    fs::create_dir(&model_directory).expect("the synthetic model directory should be creatable");
    let source_path = model_directory.join("source.bin");
    let preparer = AlignedExpertPackPreparer::from_layer_plans(
        &model_directory,
        "synthetic-model",
        "revision-1",
        vec![write_synthetic_expert_source_for_layer(&source_path, 0)],
    )
    .expect("the synthetic preparation should plan");
    let invalid_final_revision = model_directory
        .join(".astronomical-aligned-expert-packs")
        .join("revision-1");
    fs::create_dir_all(&invalid_final_revision)
        .expect("the invalid generated revision should be creatable");

    let non_replacing_outcome = preparer.prepare(false, |_progress_event| {});
    assert!(matches!(
        non_replacing_outcome,
        Err(AlignedExpertPackPreparationError::InvalidExistingRevision { .. })
    ));

    let replacement_report = preparer
        .prepare(true, |_progress_event| {})
        .expect("explicit replacement should rebuild the generated revision");
    assert!(!replacement_report.reused_existing_pack_set);
    assert!(replacement_report.final_revision_directory.is_dir());
}
