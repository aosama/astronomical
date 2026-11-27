//! Whether the engine is fit to be measured, and which build it is, read from
//! its status. Speed is measured only on Development, with the
//! `astro_ssd_prompt_cache` and the skill declaration off: every task reports
//! prompt reading, which AGENTS.md measures with the `astro_ssd_prompt_cache`
//! off, since a restored prompt would skip it; a release ledger also needs a clean
//! build, so its figures belong to one commit.

use serde_json::Value;

const STATUS_PREFIX: &str = "[measure-speed]";

/// The build the engine runs, and the memory setting to restore after the
/// reference cases change it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineBuild {
    pub version: String,
    pub commit: String,
    pub is_dirty: bool,
    pub maximum_mlx_memory_gb: Option<u64>,
}

/// Refuses an instance other than Development, or one that would add a
/// cached prompt or the skill's note to what is measured. Settings are never
/// changed from here: the person's instance is theirs.
pub fn refuse_unless_set_up_for_measuring(status_document: &Value) -> Result<(), String> {
    if status_document["application"]["channel"] != "development" {
        return Err(format!(
            "{STATUS_PREFIX} status=refused reason=not-development: measure against the Development instance only"
        ));
    }
    let prompt_cache_enabled = is_prompt_cache_enabled(
        &status_document["configuration"]["astro_ssd_prompt_cache"]["enabled"],
    );
    let skill_declared = status_document["skill"]["declared_to_models"]
        .as_bool()
        .unwrap_or(true);
    if prompt_cache_enabled || skill_declared {
        return Err(format!(
            "{STATUS_PREFIX} status=refused reason=engine-not-set-up: set \"ssd_prompt_cache\": {{\"enabled\": false}} and \"skill\": {{\"declare_to_models\": false}} in the engine's config.json, restart it, and run again (prompt cache on: {prompt_cache_enabled}, skill declared: {skill_declared})"
        ));
    }
    Ok(())
}

/// Whether the `astro_ssd_prompt_cache` will serve the measured replies. The
/// effective value is null until a model loads, and a measurement starts with
/// none loaded, so the configured value stands in, then the default; a
/// setting reported nowhere counts as on.
fn is_prompt_cache_enabled(enabled_setting: &Value) -> bool {
    ["effective", "configured", "default"]
        .iter()
        .find_map(|source| enabled_setting[*source].as_bool())
        .unwrap_or(true)
}

/// The engine's version and commit; a build whose cleanliness is not
/// reported is taken as dirty, so it can never write a release ledger.
pub fn engine_build_from_status(status_document: &Value) -> Result<EngineBuild, String> {
    let application_facts = &status_document["application"];
    let version = application_facts["version"]
        .as_str()
        .ok_or_else(|| format!("{STATUS_PREFIX} status=refused reason=engine-version-unknown"))?;
    Ok(EngineBuild {
        version: version.to_owned(),
        commit: application_facts["commit"]
            .as_str()
            .unwrap_or("unknown")
            .to_owned(),
        is_dirty: application_facts["is_dirty"].as_bool().unwrap_or(true),
        maximum_mlx_memory_gb: status_document["configured_maximum_mlx_memory_gb"].as_u64(),
    })
}

/// A release ledger's figures must belong to one commit.
pub fn refuse_dirty_build_for_ledger(
    should_write_ledger: bool,
    engine_build: &EngineBuild,
) -> Result<(), String> {
    if should_write_ledger && engine_build.is_dirty {
        return Err(format!(
            "{STATUS_PREFIX} status=refused reason=engine-build-dirty: a release ledger is measured on a clean build of a commit"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "engine_setup_tests.rs"]
mod engine_setup_tests;
