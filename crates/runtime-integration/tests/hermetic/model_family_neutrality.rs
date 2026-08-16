use std::fs;
use std::path::PathBuf;

#[test]
fn should_keep_every_runtime_source_free_of_concrete_model_families() {
    let source_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut pending_directories = vec![source_directory];
    let mut violations = Vec::new();
    while let Some(pending_directory) = pending_directories.pop() {
        for entry in fs::read_dir(&pending_directory).expect("runtime source should be readable") {
            let entry_path = entry
                .expect("runtime source entry should be readable")
                .path();
            if entry_path.is_dir() {
                pending_directories.push(entry_path);
                continue;
            }
            if entry_path
                .extension()
                .and_then(|extension| extension.to_str())
                != Some("rs")
            {
                continue;
            }
            let source =
                fs::read_to_string(&entry_path).expect("runtime source should be readable");
            for family_identifier in [
                "qwen3_5",
                "Qwen3_5",
                "laguna",
                "Laguna",
                "deepseek_v4",
                "DeepSeekV4",
            ] {
                if source.contains(family_identifier) {
                    violations.push((entry_path.clone(), family_identifier));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "runtime integration must remain family-neutral: {violations:#?}"
    );
}
