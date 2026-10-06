use astronomical_model_serving::{
    TensorDeclarationOrigin, TensorDtype, TensorInventory, TensorInventoryError, TensorLocation,
    TensorProfile, TensorSemanticRole, TensorSourceId,
    validate_safetensors_required_profiles_for_tests,
};
use std::fs;

fn target_location(
    canonical_name: &str,
    stored_name: &str,
    source_id: TensorSourceId,
    declaration_origin: TensorDeclarationOrigin,
) -> TensorLocation {
    TensorLocation::new(
        canonical_name,
        stored_name,
        source_id,
        TensorSemanticRole::Target,
        declaration_origin,
    )
}

#[test]
fn should_resolve_canonical_target_names_to_stored_source_names() {
    let source_id = TensorSourceId::new(7);
    let mut inventory = TensorInventory::new();
    inventory
        .insert(target_location(
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            source_id,
            TensorDeclarationOrigin::MainIndex,
        ))
        .expect("the unique target tensor should enter the inventory");

    let location = inventory
        .location("language_model.model.layers.0.mlp.gate_proj.weight")
        .expect("the canonical target tensor should resolve");
    assert_eq!(
        location.stored_name(),
        "model.layers.0.mlp.gate_proj.weight"
    );
    assert_eq!(location.source_id(), source_id);
}

#[test]
fn should_reject_embedded_and_sidecar_canonical_collisions() {
    let mut inventory = TensorInventory::new();
    inventory
        .insert(target_location(
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "language_model.model.layers.0.mlp.gate_proj.weight",
            TensorSourceId::new(1),
            TensorDeclarationOrigin::MainIndex,
        ))
        .expect("the embedded location should enter the inventory");

    let collision = inventory
        .insert(target_location(
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            TensorSourceId::new(2),
            TensorDeclarationOrigin::ArchitectureSidecar,
        ))
        .expect_err("the sidecar must not silently override the indexed tensor");

    assert!(matches!(
        collision,
        TensorInventoryError::CanonicalNameCollision { canonical_name }
            if canonical_name == "language_model.model.layers.0.mlp.gate_proj.weight"
    ));
}

#[test]
fn should_reject_duplicate_physical_tensor_locations() {
    let source_id = TensorSourceId::new(3);
    let mut inventory = TensorInventory::new();
    inventory
        .insert(target_location(
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            source_id,
            TensorDeclarationOrigin::MainIndex,
        ))
        .expect("the first physical location should enter the inventory");

    let duplicate = inventory
        .insert(target_location(
            "language_model.model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            source_id,
            TensorDeclarationOrigin::MainIndex,
        ))
        .expect_err("one physical tensor must not have two canonical identities");

    assert!(matches!(
        duplicate,
        TensorInventoryError::PhysicalLocationCollision { stored_name, .. }
            if stored_name == "model.layers.0.mlp.gate_proj.weight"
    ));
}

#[test]
fn should_reject_a_wrong_dtype_on_a_required_target_tensor() {
    let model_directory = tempfile::tempdir().expect("the synthetic model directory should exist");
    let source_id = TensorSourceId::new(1);
    let mut inventory = TensorInventory::new();
    inventory
        .insert(target_location(
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "language_model.model.layers.0.mlp.gate_proj.weight",
            source_id,
            TensorDeclarationOrigin::MainIndex,
        ))
        .expect("the required target tensor should enter the inventory");

    // The physical source is structurally valid, but the required target dtype
    // conflicts with its canonical profile, so validation must reject the source.
    let header = r#"{"language_model.model.layers.0.mlp.gate_proj.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
    let mut source_bytes = Vec::new();
    source_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    source_bytes.extend_from_slice(header.as_bytes());
    source_bytes.extend_from_slice(&[0_u8; 2]);
    fs::write(
        model_directory.path().join("model.safetensors"),
        source_bytes,
    )
    .expect("the synthetic target source should be written");
    let profiles = vec![TensorProfile {
        name: "language_model.model.layers.0.mlp.gate_proj.weight".to_owned(),
        dtype: TensorDtype::Float32,
        shape: vec![1],
        equivalent_published_shapes: Vec::new(),
    }];

    let validation_outcome = validate_safetensors_required_profiles_for_tests(
        model_directory.path(),
        "model.safetensors",
        &inventory,
        &profiles,
    );

    assert!(
        validation_outcome.is_err(),
        "a required target dtype mismatch must reject the source"
    );
}

#[test]
fn should_accept_a_declared_published_shape_through_the_retained_source_validator() {
    let model_directory = tempfile::tempdir().expect("the synthetic model directory should exist");
    let source_id = TensorSourceId::new(1);
    let mut inventory = TensorInventory::new();
    inventory
        .insert(TensorLocation::new(
            "vision_tower.patch_embed.proj.weight",
            "vision_tower.patch_embed.proj.weight",
            source_id,
            TensorSemanticRole::Vision,
            TensorDeclarationOrigin::MainIndex,
        ))
        .expect("the vision patch embedding should enter the source inventory");

    let header = r#"{"vision_tower.patch_embed.proj.weight":{"dtype":"F32","shape":[2,3,2,1,1],"data_offsets":[0,48]}}"#;
    let mut source_bytes = Vec::new();
    source_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    source_bytes.extend_from_slice(header.as_bytes());
    source_bytes.extend_from_slice(&[0_u8; 48]);
    fs::write(
        model_directory.path().join("vision.safetensors"),
        source_bytes,
    )
    .expect("the synthetic vision source should be written");

    let profiles = [TensorProfile {
        name: "vision_tower.patch_embed.proj.weight".to_owned(),
        dtype: TensorDtype::Float32,
        shape: vec![2, 2, 1, 1, 3],
        equivalent_published_shapes: vec![vec![2, 3, 2, 1, 1]],
    }];
    let validation_outcome = validate_safetensors_required_profiles_for_tests(
        model_directory.path(),
        "vision.safetensors",
        &inventory,
        &profiles,
    );

    assert!(
        validation_outcome.is_ok(),
        "the retained source validator should accept the declared published permutation"
    );
}
