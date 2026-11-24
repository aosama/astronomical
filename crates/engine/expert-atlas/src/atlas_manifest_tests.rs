use super::{
    ATLAS_FORMAT_NAME, ATLAS_FORMAT_VERSION, ATLAS_MANIFEST_FILE_NAME, AtlasDeclaredFile,
    AtlasExpertLayer, AtlasManifest, AtlasManifestError, AtlasRecordSlot, AtlasSource,
    declares_atlas_manifest, expert_slice_byte_count,
};

fn two_slot_layer() -> AtlasExpertLayer {
    AtlasExpertLayer {
        layer_index: 5,
        file_name: "experts/layer-005.atlas".to_owned(),
        expert_count: 4,
        record_byte_count: 1_100,
        record_stride_bytes: 16_384,
        slots: vec![
            AtlasRecordSlot {
                tensor_name: "gate.weight".to_owned(),
                dtype: "U32".to_owned(),
                shape: vec![4, 250],
                offset_in_record: 0,
                bytes_per_expert: 1_000,
            },
            AtlasRecordSlot {
                tensor_name: "gate.scales".to_owned(),
                dtype: "BF16".to_owned(),
                shape: vec![4, 50],
                offset_in_record: 1_000,
                bytes_per_expert: 100,
            },
        ],
    }
}

fn manifest_with(expert_layer: AtlasExpertLayer) -> AtlasManifest {
    AtlasManifest {
        format: ATLAS_FORMAT_NAME.to_owned(),
        format_version: ATLAS_FORMAT_VERSION,
        source: AtlasSource {
            model_id: "example/model".to_owned(),
            revision: None,
        },
        resident_files: Vec::new(),
        expert_files: vec![AtlasDeclaredFile {
            file_name: expert_layer.file_name.clone(),
            expected_file_byte_count: expert_layer.file_byte_count(),
            sha256: String::new(),
        }],
        expert_layers: vec![expert_layer],
    }
}

fn write_manifest(atlas_manifest: &AtlasManifest) -> tempfile::TempDir {
    let model_directory = tempfile::tempdir().expect("scratch");
    std::fs::write(
        model_directory.path().join(ATLAS_MANIFEST_FILE_NAME),
        serde_json::to_vec(atlas_manifest).expect("encodes"),
    )
    .expect("writes");
    model_directory
}

#[test]
fn should_read_back_a_manifest_it_wrote() {
    let atlas_manifest = manifest_with(two_slot_layer());
    let model_directory = write_manifest(&atlas_manifest);
    assert_eq!(
        AtlasManifest::read_from_directory(model_directory.path()).expect("reads"),
        atlas_manifest
    );
    assert!(declares_atlas_manifest(model_directory.path()));
}

#[test]
fn should_locate_an_expert_slot_by_record_stride_and_slot_offset() {
    let expert_layer = two_slot_layer();
    assert_eq!(expert_layer.slot_file_offset(0, &expert_layer.slots[0]), 0);
    assert_eq!(
        expert_layer.slot_file_offset(3, &expert_layer.slots[1]),
        3 * 16_384 + 1_000
    );
    assert_eq!(expert_layer.file_byte_count(), 4 * 16_384);
}

#[test]
fn should_refuse_the_retired_version_3_layout() {
    let mut atlas_manifest = manifest_with(two_slot_layer());
    atlas_manifest.format_version = 3;
    let model_directory = write_manifest(&atlas_manifest);
    assert!(matches!(
        AtlasManifest::read_from_directory(model_directory.path()),
        Err(AtlasManifestError::NotAnAtlas { .. })
    ));
    assert!(!declares_atlas_manifest(model_directory.path()));
}

#[test]
fn should_refuse_slots_with_a_gap_between_them() {
    let mut gapped_layer = two_slot_layer();
    gapped_layer.slots[1].offset_in_record = 1_004;
    let model_directory = write_manifest(&manifest_with(gapped_layer));
    assert!(matches!(
        AtlasManifest::read_from_directory(model_directory.path()),
        Err(AtlasManifestError::InconsistentLayer { layer_index: 5, .. })
    ));
}

#[test]
fn should_refuse_a_stride_that_is_not_the_record_rounded_to_a_page() {
    let mut unaligned_layer = two_slot_layer();
    unaligned_layer.record_stride_bytes = 1_100;
    let model_directory = write_manifest(&manifest_with(unaligned_layer));
    assert!(matches!(
        AtlasManifest::read_from_directory(model_directory.path()),
        Err(AtlasManifestError::InconsistentLayer { .. })
    ));
}

#[test]
fn should_refuse_a_slot_whose_shape_does_not_lead_with_the_expert_count() {
    let mut misshapen_layer = two_slot_layer();
    misshapen_layer.slots[0].shape = vec![8, 250];
    let model_directory = write_manifest(&manifest_with(misshapen_layer));
    assert!(matches!(
        AtlasManifest::read_from_directory(model_directory.path()),
        Err(AtlasManifestError::InconsistentLayer { .. })
    ));
}

#[test]
fn should_report_a_missing_manifest_as_unreadable() {
    let model_directory = tempfile::tempdir().expect("scratch");
    assert!(matches!(
        AtlasManifest::read_from_directory(model_directory.path()),
        Err(AtlasManifestError::Unreadable { .. })
    ));
    assert!(!declares_atlas_manifest(model_directory.path()));
}

fn read_back(atlas_manifest: &AtlasManifest) -> Result<AtlasManifest, AtlasManifestError> {
    let model_directory = write_manifest(atlas_manifest);
    AtlasManifest::read_from_directory(model_directory.path())
}

#[test]
fn should_size_an_expert_slice_from_its_dtype_and_shape() {
    let expert_layer = two_slot_layer();
    assert_eq!(expert_slice_byte_count(&expert_layer.slots[0]), Some(1_000));
    assert_eq!(expert_slice_byte_count(&expert_layer.slots[1]), Some(100));
}

#[test]
fn should_size_a_one_dimensional_slot_as_one_element_per_expert() {
    let mut scalar_slot = two_slot_layer().slots[1].clone();
    scalar_slot.shape = vec![4];
    assert_eq!(expert_slice_byte_count(&scalar_slot), Some(2));
}

#[test]
fn should_refuse_a_slot_whose_bytes_disagree_with_its_shape() {
    let mut mislabelled_layer = two_slot_layer();
    mislabelled_layer.slots[1].dtype = "F32".to_owned();
    assert!(matches!(
        read_back(&manifest_with(mislabelled_layer)),
        Err(AtlasManifestError::InconsistentLayer { layer_index: 5, .. })
    ));
}

#[test]
fn should_refuse_a_slot_with_an_unknown_dtype() {
    let mut unknown_dtype_layer = two_slot_layer();
    unknown_dtype_layer.slots[0].dtype = "Q4".to_owned();
    assert_eq!(expert_slice_byte_count(&unknown_dtype_layer.slots[0]), None);
    assert!(read_back(&manifest_with(unknown_dtype_layer)).is_err());
}

#[test]
fn should_refuse_a_layer_file_that_is_not_declared() {
    let mut atlas_manifest = manifest_with(two_slot_layer());
    atlas_manifest.expert_files.clear();
    assert!(matches!(
        read_back(&atlas_manifest),
        Err(AtlasManifestError::InconsistentFiles { .. })
    ));
}

#[test]
fn should_refuse_a_declared_layer_size_that_is_not_its_records() {
    let mut atlas_manifest = manifest_with(two_slot_layer());
    atlas_manifest.expert_files[0].expected_file_byte_count -= 1;
    assert!(matches!(
        read_back(&atlas_manifest),
        Err(AtlasManifestError::InconsistentFiles { .. })
    ));
}

#[test]
fn should_refuse_file_names_that_leave_the_model_directory() {
    for escaping_file_name in [
        "../outside.atlas",
        "/etc/outside.atlas",
        "experts/../../x",
        "",
    ] {
        let mut escaping_layer = two_slot_layer();
        escaping_layer.file_name = escaping_file_name.to_owned();
        assert!(
            matches!(
                read_back(&manifest_with(escaping_layer)),
                Err(AtlasManifestError::InconsistentFiles { .. })
            ),
            "{escaping_file_name:?}"
        );
    }
}
