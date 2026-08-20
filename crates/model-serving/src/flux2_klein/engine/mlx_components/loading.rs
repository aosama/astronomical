//! Validated artifact loading, memory admission, and load-time attribution.

use super::*;

const MAXIMUM_LOAD_FAILURE_DESCRIPTION_CHARACTERS: usize = 512;

impl Flux2KleinMlxComponents {
    pub(super) fn load_inner(&mut self) -> Result<Flux2KleinComponentLoad, String> {
        let mut attribution = self.new_attribution();
        let mut attribution_log = PerformanceAttributionLog::open(
            &self.performance_attribution_log_path,
            self.performance_attribution_enabled,
        )
        .map_err(|error| error.to_string())?;
        let mut runtime = None;
        let mut total_artifact_payload_bytes = None;
        let mut model_shard_count = None;
        let load_result = (|| {
            let memory_limits = MlxMemoryLimits::new(
                self.effective_mlx_memory_ceiling_bytes,
                self.allocator_cache_memory_limit_bytes,
            )
            .map_err(|error| error.to_string())?;
            let initialized_runtime = attribution
                .measure_operation(PerformanceOperation::ImageRuntimeSetup, |_| {
                    MlxRuntime::initialize(memory_limits)
                })
                .map_err(|error| error.to_string())?;
            attribution
                .measure_operation(PerformanceOperation::MlxAllocatorCacheCleanup, |_| {
                    initialized_runtime.clear_allocator_cache()
                })
                .map_err(|error| error.to_string())?;
            runtime = Some(initialized_runtime);
            let validated_artifact = Flux2KleinArtifactValidator::new()
                .validate_with_performance_attribution(
                    &self.model_directory,
                    self.provenance.clone(),
                    &mut attribution,
                )
                .map_err(|error| error.to_string())?;
            let geometry = memory_geometry(&validated_artifact)?;
            total_artifact_payload_bytes = Some(
                geometry
                    .text_encoder_payload_bytes
                    .checked_add(validated_artifact.transformer_inventory().payload_bytes())
                    .and_then(|payload_bytes| {
                        payload_bytes
                            .checked_add(validated_artifact.vae_inventory().payload_bytes())
                    })
                    .ok_or_else(|| {
                        "FLUX.2 Klein artifact payload accounting overflowed".to_owned()
                    })?,
            );
            model_shard_count = Some(
                validated_artifact
                    .text_shard_count()
                    .checked_add(
                        validated_artifact
                            .transformer_inventory()
                            .source_file_count(),
                    )
                    .and_then(|shard_count| {
                        shard_count
                            .checked_add(validated_artifact.vae_inventory().source_file_count())
                    })
                    .ok_or_else(|| {
                        "FLUX.2 Klein artifact shard accounting overflowed".to_owned()
                    })?,
            );
            let transformer_geometry =
                super::super::super::Flux2KleinTransformerGeometry::from_config(
                    validated_artifact.transformer_config(),
                )
                .map_err(|error| error.to_string())?;
            let residency_plan = attribution
                .measure_operation(PerformanceOperation::MemoryAdmissionSnapshot, |_| {
                    Flux2KleinMemoryAdmission::plan(
                        self.effective_mlx_memory_ceiling_bytes as u64,
                        &geometry,
                    )
                })
                .map_err(|error| error.to_string())?;
            let minimum_ceiling_bytes = residency_plan.minimum_mlx_memory_ceiling_bytes();
            Ok::<_, String>((
                validated_artifact,
                residency_plan,
                transformer_geometry,
                minimum_ceiling_bytes,
            ))
        })();
        let memory_snapshot = runtime
            .as_ref()
            .and_then(|initialized_runtime| initialized_runtime.memory_snapshot().ok());
        let failure_description = load_result.as_ref().err().map(|failure| {
            failure
                .chars()
                .take(MAXIMUM_LOAD_FAILURE_DESCRIPTION_CHARACTERS)
                .collect::<String>()
        });
        let outcome = if load_result.is_ok() {
            PerformanceAttributionOutcome::Success
        } else {
            PerformanceAttributionOutcome::Failed
        };
        if let Some(report) =
            attribution.finish_model_loading(ModelLoadingPerformanceAttributionMetadata {
                outcome,
                model_id: Some(self.serving_model_id.clone()),
                model_revision: Some(FLUX2_KLEIN_OFFICIAL_REVISION.to_owned()),
                prefill_transient_observation_completed: false,
                prefill_observed_transient_high_water_bytes: 0,
                total_artifact_payload_bytes,
                resident_model_payload_bytes: Some(0),
                model_shard_count,
                mlx_active_memory_bytes: memory_snapshot
                    .map(|snapshot| snapshot.active_memory_bytes() as u64),
                mlx_allocator_cache_memory_bytes: memory_snapshot
                    .map(|snapshot| snapshot.allocator_cache_memory_bytes() as u64),
                mlx_peak_memory_bytes: memory_snapshot
                    .map(|snapshot| snapshot.peak_memory_bytes() as u64),
                failure_description,
            })
        {
            attribution_log
                .record(&report)
                .map_err(|error| error.to_string())?;
        }
        let (validated_artifact, residency_plan, transformer_geometry, minimum_ceiling_bytes) =
            load_result?;
        self.runtime = runtime;
        self.validated_artifact = Some(validated_artifact);
        self.residency_plan = Some(residency_plan);
        self.transformer_geometry = Some(transformer_geometry);
        self.performance_attribution_log = Some(attribution_log);
        Ok(Flux2KleinComponentLoad::new(
            self.serving_model_id.clone(),
            FLUX2_KLEIN_OFFICIAL_REVISION,
            official_capabilities(),
            minimum_ceiling_bytes,
        ))
    }
}
