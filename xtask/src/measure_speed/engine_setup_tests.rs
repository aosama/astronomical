use serde_json::json;

use super::{
    EngineBuild, engine_build_from_status, refuse_dirty_build_for_ledger,
    refuse_unless_set_up_for_measuring,
};

fn development_status_set_up_for_measuring() -> serde_json::Value {
    json!({
        "application": {"channel": "development", "version": "0.2.134", "commit": "a245a15d", "is_dirty": false},
        "configuration": {"astro_ssd_prompt_cache": {"enabled": {"effective": false}}},
        "skill": {"declared_to_models": false},
        "configured_maximum_mlx_memory_gb": null
    })
}

#[test]
fn should_accept_development_with_the_prompt_cache_and_skill_note_off() {
    let status_document = development_status_set_up_for_measuring();

    assert_eq!(refuse_unless_set_up_for_measuring(&status_document), Ok(()));
    assert_eq!(
        engine_build_from_status(&status_document),
        Ok(EngineBuild {
            version: "0.2.134".to_owned(),
            commit: "a245a15d".to_owned(),
            is_dirty: false,
            maximum_mlx_memory_gb: None,
        })
    );
}

#[test]
fn should_refuse_stable_a_prompt_cache_a_skill_note_or_unknown_settings() {
    let mut stable = development_status_set_up_for_measuring();
    stable["application"]["channel"] = json!("stable");
    let mut prompt_cache_on = development_status_set_up_for_measuring();
    prompt_cache_on["configuration"]["astro_ssd_prompt_cache"]["enabled"]["effective"] =
        json!(true);
    let mut skill_declared = development_status_set_up_for_measuring();
    skill_declared["skill"]["declared_to_models"] = json!(true);
    let mut settings_unreported = development_status_set_up_for_measuring();
    settings_unreported["skill"] = json!(null);

    for refused_status in [stable, prompt_cache_on, skill_declared, settings_unreported] {
        assert!(refuse_unless_set_up_for_measuring(&refused_status).is_err());
    }
}

#[test]
fn should_take_a_build_of_unknown_cleanliness_as_dirty_and_refuse_no_version() {
    let mut cleanliness_unreported = development_status_set_up_for_measuring();
    cleanliness_unreported["application"]["is_dirty"] = json!(null);
    let mut version_unreported = development_status_set_up_for_measuring();
    version_unreported["application"]["version"] = json!(null);

    assert!(
        engine_build_from_status(&cleanliness_unreported)
            .expect("a build with a version")
            .is_dirty
    );
    assert!(engine_build_from_status(&version_unreported).is_err());
}

#[test]
fn should_refuse_a_ledger_only_from_a_dirty_build() {
    let clean_build = engine_build_from_status(&development_status_set_up_for_measuring())
        .expect("a clean build");
    let dirty_build = EngineBuild {
        is_dirty: true,
        ..clean_build.clone()
    };

    assert_eq!(refuse_dirty_build_for_ledger(true, &clean_build), Ok(()));
    assert_eq!(refuse_dirty_build_for_ledger(false, &dirty_build), Ok(()));
    assert!(refuse_dirty_build_for_ledger(true, &dirty_build).is_err());
}
