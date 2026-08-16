use astronomical_model_serving::{
    TensorDeclarationOrigin, TensorDtype, TensorFeature, TensorInventory, TensorInventoryError,
    TensorLocation, TensorProfile, TensorSemanticRole, TensorSourceId,
    validate_safetensors_profile_partitions_for_tests,
};
use std::fs;

fn mtp_location(
    canonical_name: &str,
    stored_name: &str,
    source_id: TensorSourceId,
    declaration_origin: TensorDeclarationOrigin,
) -> TensorLocation {
    TensorLocation::new(
        canonical_name,
        stored_name,
        source_id,
        TensorSemanticRole::MultiTokenPrediction,
        declaration_origin,
        Some(TensorFeature::MultiTokenPrediction),
    )
}

#[test]
fn should_resolve_canonical_mtp_names_to_stored_sidecar_names() {
    let source_id = TensorSourceId::new(7);
    let mut inventory = TensorInventory::new();
    inventory
        .insert(mtp_location(
            "language_model.mtp.fc.weight",
            "mtp.fc.weight",
            source_id,
            TensorDeclarationOrigin::ArchitectureSidecar,
        ))
        .expect("the unique sidecar tensor should enter the inventory");

    let location = inventory
        .location("language_model.mtp.fc.weight")
        .expect("the canonical MTP tensor should resolve");
    assert_eq!(location.stored_name(), "mtp.fc.weight");
    assert_eq!(location.source_id(), source_id);
}

#[test]
fn should_reject_embedded_and_sidecar_canonical_collisions() {
    let mut inventory = TensorInventory::new();
    inventory
        .insert(mtp_location(
            "language_model.mtp.fc.weight",
            "language_model.mtp.fc.weight",
            TensorSourceId::new(1),
            TensorDeclarationOrigin::MainIndex,
        ))
        .expect("the embedded location should enter the inventory");

    let collision = inventory
        .insert(mtp_location(
            "language_model.mtp.fc.weight",
            "mtp.fc.weight",
            TensorSourceId::new(2),
            TensorDeclarationOrigin::ArchitectureSidecar,
        ))
        .expect_err("the sidecar must not silently override embedded MTP");

    assert!(matches!(
        collision,
        TensorInventoryError::CanonicalNameCollision { canonical_name }
            if canonical_name == "language_model.mtp.fc.weight"
    ));
}

#[test]
fn should_reject_duplicate_physical_tensor_locations() {
    let source_id = TensorSourceId::new(3);
    let mut inventory = TensorInventory::new();
    inventory
        .insert(mtp_location(
            "language_model.mtp.fc.weight",
            "mtp.fc.weight",
            source_id,
            TensorDeclarationOrigin::ArchitectureSidecar,
        ))
        .expect("the first physical location should enter the inventory");

    let duplicate = inventory
        .insert(mtp_location(
            "language_model.mtp.alias.weight",
            "mtp.fc.weight",
            source_id,
            TensorDeclarationOrigin::ArchitectureSidecar,
        ))
        .expect_err("one physical tensor must not have two canonical identities");

    assert!(matches!(
        duplicate,
        TensorInventoryError::PhysicalLocationCollision { stored_name, .. }
            if stored_name == "mtp.fc.weight"
    ));
}

#[test]
fn should_remove_the_complete_optional_mtp_feature_after_a_collision() {
    let source_id = TensorSourceId::new(9);
    let mut inventory = TensorInventory::new();
    for tensor_suffix in ["weight", "scales", "biases"] {
        inventory
            .insert(mtp_location(
                &format!("language_model.mtp.proj.{tensor_suffix}"),
                &format!("mtp.proj.{tensor_suffix}"),
                source_id,
                TensorDeclarationOrigin::ArchitectureSidecar,
            ))
            .expect("the quantization companion should enter the inventory");
    }

    inventory.remove_feature(TensorFeature::MultiTokenPrediction);

    assert_eq!(inventory.tensor_count(), 0);
    assert!(inventory.source_ids().next().is_none());
}

#[test]
fn should_preserve_required_target_profiles_when_embedded_optional_mtp_has_the_wrong_dtype() {
    let model_directory = tempfile::tempdir().expect("the synthetic model directory should exist");
    let source_id = TensorSourceId::new(1);
    let mut inventory = TensorInventory::new();
    inventory
        .insert(TensorLocation::new(
            "language_model.target.weight",
            "language_model.target.weight",
            source_id,
            TensorSemanticRole::Target,
            TensorDeclarationOrigin::MainIndex,
            None,
        ))
        .expect("the required target tensor should enter the inventory");
    inventory
        .insert(mtp_location(
            "language_model.mtp.proj.weight",
            "language_model.mtp.proj.weight",
            source_id,
            TensorDeclarationOrigin::MainIndex,
        ))
        .expect("the embedded optional tensor should enter the same source inventory");

    // The physical source is structurally valid and its target tensor matches. Only the optional
    // MTP dtype conflicts with its canonical profile, so target serving must remain available.
    let header = r#"{"language_model.target.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"language_model.mtp.proj.weight":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#;
    let mut source_bytes = Vec::new();
    source_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    source_bytes.extend_from_slice(header.as_bytes());
    source_bytes.extend_from_slice(&[0_u8; 4]);
    fs::write(
        model_directory.path().join("model.safetensors"),
        source_bytes,
    )
    .expect("the synthetic shared target and MTP source should be written");
    let profiles = vec![
        TensorProfile {
            name: "language_model.target.weight".to_owned(),
            dtype: TensorDtype::BFloat16,
            shape: vec![1],
        },
        TensorProfile {
            name: "language_model.mtp.proj.weight".to_owned(),
            dtype: TensorDtype::UInt32,
            shape: vec![1],
        },
    ];

    let optional_mtp_profiles_are_valid = validate_safetensors_profile_partitions_for_tests(
        model_directory.path(),
        "model.safetensors",
        &inventory,
        &profiles,
        TensorFeature::MultiTokenPrediction,
    )
    .expect("the required target profile should remain valid");

    assert!(!optional_mtp_profiles_are_valid);
}

#[test]
fn should_preserve_required_target_when_optional_mtp_uses_a_known_unsupported_dtype() {
    let model_directory = tempfile::tempdir().expect("the synthetic model directory should exist");
    let source_id = TensorSourceId::new(1);
    let mut inventory = TensorInventory::new();
    inventory
        .insert(TensorLocation::new(
            "language_model.target.weight",
            "language_model.target.weight",
            source_id,
            TensorSemanticRole::Target,
            TensorDeclarationOrigin::MainIndex,
            None,
        ))
        .expect("the required target tensor should enter the inventory");
    inventory
        .insert(mtp_location(
            "language_model.mtp.proj.weight",
            "language_model.mtp.proj.weight",
            source_id,
            TensorDeclarationOrigin::MainIndex,
        ))
        .expect("the optional MTP tensor should enter the shared source inventory");

    // U16 is structurally valid SafeTensors storage but unsupported by this MTP execution
    // profile. Structural parsing must succeed so the optional feature can be disabled without
    // rejecting the valid target tensor that shares this physical source.
    let header = r#"{"language_model.target.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"language_model.mtp.proj.weight":{"dtype":"U16","shape":[1],"data_offsets":[2,4]}}"#;
    let mut source_bytes = Vec::new();
    source_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    source_bytes.extend_from_slice(header.as_bytes());
    source_bytes.extend_from_slice(&[0_u8; 4]);
    fs::write(
        model_directory.path().join("model.safetensors"),
        source_bytes,
    )
    .expect("the synthetic shared target and MTP source should be written");
    let profiles = vec![
        TensorProfile {
            name: "language_model.target.weight".to_owned(),
            dtype: TensorDtype::BFloat16,
            shape: vec![1],
        },
        TensorProfile {
            name: "language_model.mtp.proj.weight".to_owned(),
            dtype: TensorDtype::UInt32,
            shape: vec![1],
        },
    ];

    let optional_mtp_profiles_are_valid = validate_safetensors_profile_partitions_for_tests(
        model_directory.path(),
        "model.safetensors",
        &inventory,
        &profiles,
        TensorFeature::MultiTokenPrediction,
    )
    .expect("the required target profile should remain valid");

    assert!(!optional_mtp_profiles_are_valid);
}
