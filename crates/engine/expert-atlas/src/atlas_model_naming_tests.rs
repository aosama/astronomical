use super::{ATLAS_MODEL_NAME_SUFFIX, ModelIdError, atlas_model_id_for, organization_and_name};

#[test]
fn should_append_the_atlas_suffix_to_an_organization_and_name() {
    assert_eq!(
        atlas_model_id_for("ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit"),
        "ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit-astro-atlas-streaming"
    );
}

#[test]
fn should_not_append_the_suffix_twice() {
    let atlas_model_id = format!("organization/model{ATLAS_MODEL_NAME_SUFFIX}");
    assert_eq!(atlas_model_id_for(&atlas_model_id), atlas_model_id);
}

#[test]
fn should_split_a_plain_organization_and_name() {
    assert_eq!(
        organization_and_name("organization/model-name"),
        Ok(("organization", "model-name"))
    );
}

#[test]
fn should_refuse_an_id_without_an_organization() {
    assert_eq!(
        organization_and_name("model-name"),
        Err(ModelIdError::NotOrganizationAndName(
            "model-name".to_owned()
        ))
    );
}

#[test]
fn should_refuse_components_that_could_leave_the_library() {
    for unsafe_model_id in [
        "../model",
        "organization/..",
        "organization/",
        "/model",
        "organization/a/b",
        "organization/.",
    ] {
        assert!(
            organization_and_name(unsafe_model_id).is_err(),
            "{unsafe_model_id} must be refused"
        );
    }
}
