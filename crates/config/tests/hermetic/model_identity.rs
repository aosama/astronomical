use astronomical_config::near_model_matches;

#[test]
fn near_model_matches_survive_partial_typos() {
    let candidates = &["Qwen3.5-2B-4bit", "Ornith-1.5-35B-6bit", "gpt-oss-20b"];
    assert_eq!(
        near_model_matches("qwen3.5-2b", candidates),
        vec!["qwen3.5-2b-4bit".to_owned()]
    );
    assert_eq!(
        near_model_matches("qwen3-5-2b-4bit", candidates),
        vec!["qwen3.5-2b-4bit".to_owned()]
    );
    assert_eq!(
        near_model_matches("other-ns/Qwen3.5-2B-4bit", candidates),
        vec!["qwen3.5-2b-4bit".to_owned()]
    );
    assert!(near_model_matches("definitely-not-a-model", candidates).is_empty());
}

#[test]
fn near_model_matches_accepts_owned_candidate_strings() {
    let candidates = vec!["Qwen3.5-2B-4bit".to_owned()];
    assert_eq!(
        near_model_matches("QWEN3.5", &candidates),
        vec!["qwen3.5-2b-4bit".to_owned()]
    );
}
