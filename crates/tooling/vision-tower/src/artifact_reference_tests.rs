use super::*;

fn repository(repository_id: &str, revision: Option<&str>) -> ArtifactReference {
    ArtifactReference::HuggingFaceRepository {
        repository_id: repository_id.to_owned(),
        revision: revision.map(str::to_owned),
    }
}

#[test]
fn should_read_a_repository_with_and_without_a_revision() {
    assert_eq!(
        ArtifactReference::parse("example-org/Model-1.5-MLX_4bit").unwrap(),
        repository("example-org/Model-1.5-MLX_4bit", None)
    );
    assert_eq!(
        ArtifactReference::parse("example-org/model@a48173b").unwrap(),
        repository("example-org/model", Some("a48173b"))
    );
}

#[test]
fn should_prefer_an_existing_directory_over_a_repository_reading() {
    let directory = tempfile::tempdir().unwrap();
    let directory_text = directory.path().to_str().unwrap();
    assert_eq!(
        ArtifactReference::parse(directory_text).unwrap(),
        ArtifactReference::LocalDirectory(directory.path().to_path_buf())
    );
}

#[test]
fn should_refuse_references_that_could_reshape_the_request() {
    for refused_reference in [
        "",
        "no-slash",
        "too/many/segments",
        "/organization/model",
        "organization/",
        "../model",
        "organization/..",
        "organization/model@",
        "organization/model@a/b",
        "organization/mo del",
        "organization/model?x=1",
    ] {
        assert!(
            ArtifactReference::parse(refused_reference).is_err(),
            "{refused_reference:?} should be refused"
        );
    }
}
