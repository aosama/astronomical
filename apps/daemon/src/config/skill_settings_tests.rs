use crate::config::config_reload::{ConfigReloadDecision, ConfigReloadDiff};
use crate::config::config_reload_tests::sample_resolved_config;

#[test]
fn should_apply_both_skill_switches_without_restarting_the_worker() {
    let current = sample_resolved_config();
    let mut candidate = sample_resolved_config();
    candidate.skill_settings.declared_to_models = !current.skill_settings.declared_to_models;
    candidate.skill_settings.updates_from_web = !current.skill_settings.updates_from_web;
    candidate.configuration_generation = "next-generation".to_owned();

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(matches!(
        decision,
        ConfigReloadDecision::NoWorkerRestart { ref reloaded_fields, .. }
            if reloaded_fields == &[
                "skill.declare_to_models".to_owned(),
                "skill.update_from_web".to_owned(),
            ]
    ));
}
