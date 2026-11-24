//! What the Qwen 3.5 startup opens on disk before it chooses an engine: the
//! model-loading attribution log, the model's `astro_ssd_prompt_cache` disk
//! store, and its `astro_neural_engine_program_cache` folder.

use std::path::PathBuf;

use config::{
    AstroSsdPromptCacheConfig, astro_neural_engine_program_cache_directory_for_model,
    model_weights_fingerprint, remove_other_weights_of_model,
};

use crate::{
    AstroSsdPromptCacheDiskStoreConfig, PerformanceAttribution, PerformanceAttributionLog,
};

use super::qwen3_5_model_startup_error::Qwen3_5ModelStartupError;

/// `None` keeps nothing between loads, which also covers weights that cannot be
/// identified: rebuilding is slow, and reusing another model's programs would be wrong.
pub(super) fn model_astro_neural_engine_program_cache_directory(
    neural_engine_program_directory: Option<&std::path::Path>,
    model_directory: &std::path::Path,
    model_id: &str,
    model_revision: &str,
) -> Option<PathBuf> {
    let program_directory = neural_engine_program_directory?;
    let model_program_directory = astro_neural_engine_program_cache_directory_for_model(
        program_directory,
        model_id,
        model_revision,
        &model_weights_fingerprint(model_directory)?,
    );
    remove_other_weights_of_model(program_directory, model_id, &model_program_directory);
    Some(model_program_directory)
}

/// The model-loading attribution and the log it is written to.
pub(super) fn open_model_loading_attribution(
    performance_attribution_enabled: bool,
    performance_attribution_log_path: &std::path::Path,
) -> Result<(PerformanceAttribution, PerformanceAttributionLog), Qwen3_5ModelStartupError> {
    let model_loading_performance_attribution = if performance_attribution_enabled {
        PerformanceAttribution::enabled()
    } else {
        PerformanceAttribution::disabled()
    };
    let performance_attribution_log = PerformanceAttributionLog::open(
        performance_attribution_log_path,
        performance_attribution_enabled,
    )
    .map_err(
        |source| Qwen3_5ModelStartupError::OpenPerformanceAttributionLog {
            log_path: performance_attribution_log_path.to_path_buf(),
            source,
        },
    )?;
    Ok((
        model_loading_performance_attribution,
        performance_attribution_log,
    ))
}

/// The model's `astro_ssd_prompt_cache` disk store, derived only when the user
/// enabled it. This keeps disabled operation from touching model/revision
/// identity, opening directories, scanning files, or reserving publication
/// workspace later in model loading.
pub(super) fn astro_ssd_prompt_cache_disk_store_config(
    astro_ssd_prompt_cache_enabled: bool,
    astro_ssd_prompt_cache_config: &AstroSsdPromptCacheConfig,
    model_id: &str,
    model_revision: &str,
) -> Option<AstroSsdPromptCacheDiskStoreConfig> {
    if !astro_ssd_prompt_cache_enabled {
        return None;
    }
    let per_model_astro_ssd_prompt_cache_config =
        astro_ssd_prompt_cache_config.for_model(model_id, model_revision);
    Some(AstroSsdPromptCacheDiskStoreConfig::new(
        per_model_astro_ssd_prompt_cache_config
            .active_model_astro_ssd_prompt_cache_directory()
            .clone(),
        per_model_astro_ssd_prompt_cache_config
            .global_astro_ssd_prompt_cache_root_directory()
            .clone(),
        per_model_astro_ssd_prompt_cache_config.global_astro_ssd_prompt_cache_maximum_size_bytes(),
    ))
}
