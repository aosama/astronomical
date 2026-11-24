use crate::DownloadCatalogFamily;

fn family_from_json(family_json: &str) -> Result<DownloadCatalogFamily, serde_json::Error> {
    serde_json::from_str(family_json)
}

#[test]
fn should_read_every_family_the_registry_names() {
    for family_identifier in [
        "qwen3_5",
        "prism_hadamard_qwen35",
        "flux2_klein",
        "qwen_image_21",
        "modernbert",
        "laya",
        "clef",
        "kev",
        "kokoro",
        "phonon",
    ] {
        let catalog_family = family_from_json(&format!("\"{family_identifier}\""))
            .unwrap_or_else(|error| panic!("{family_identifier} must read: {error}"));
        assert_eq!(catalog_family.as_str(), family_identifier);
    }
}

#[test]
fn should_refuse_a_family_the_registry_does_not_name() {
    let refusal = family_from_json("\"unsupported_family\"").unwrap_err();
    assert!(
        refusal
            .to_string()
            .contains("unknown model family `unsupported_family`"),
        "{refusal}"
    );
}

#[test]
fn should_refuse_an_empty_family_identifier() {
    assert!(family_from_json("\"\"").is_err());
}

#[test]
fn should_refuse_a_family_identifier_in_another_case() {
    assert!(family_from_json("\"Qwen3_5\"").is_err());
}

#[test]
fn should_refuse_a_family_that_is_not_a_string() {
    assert!(family_from_json("3").is_err());
}
