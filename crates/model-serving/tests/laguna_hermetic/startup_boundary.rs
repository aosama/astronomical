use std::fs;
use std::path::PathBuf;

#[test]
fn should_keep_laguna_startup_free_of_raw_namespace_parsing() {
    let startup_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/laguna/startup");
    let mut startup_source = String::new();
    for source_file in ["mod.rs", "weight_loader.rs", "error.rs"] {
        startup_source.push_str(
            &fs::read_to_string(startup_directory.join(source_file))
                .expect("Laguna startup source should be readable"),
        );
    }
    for forbidden_fragment in [
        "strip_prefix",
        "split('.'",
        "model.layers",
        "mlp.experts",
        "safetensors.index",
    ] {
        assert!(
            !startup_source.contains(forbidden_fragment),
            "Laguna startup must not parse raw aliases or namespaces: {forbidden_fragment}"
        );
    }
    assert!(startup_source.contains("tensor_id()"));
    assert!(startup_source.contains("raw_tensor_name()"));
}
