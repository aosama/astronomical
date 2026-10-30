//! Validated artifact loading, memory admission, and load-time attribution.

use super::*;

const MAXIMUM_LOAD_FAILURE_DESCRIPTION_CHARACTERS: usize = 512;

/// One full validated load attempt: the runtime the initialization step
/// created plus the validated plan payload only when every step succeeded.
///
/// The runtime is reported even on planning failure so the failure path can
/// still attach the memory snapshot the initialization left behind to the
/// load-time attribution report.
struct Flux2KleinValidatedLoad {
    initialized_runtime: Option<MlxRuntime>,
    plan: Result<Flux2KleinValidatedLoadPlan, String>,
}

/// Everything a successful FLUX.2 Klein load validated and planned, grouped
/// so the handoff from planning into engine state is named instead of a
/// five-element tuple.
struct Flux2KleinValidatedLoadPlan {
    validated_artifact: ValidatedFlux2KleinArtifact,
    residency_plan: Flux2KleinResidencyPlan,
    transformer_geometry: super::super::super::Flux2KleinTransformerGeometry,
    minimum_ceiling_bytes: u64,
    memory_geometry: Flux2KleinMemoryGeometry,
    total_artifact_payload_bytes: u64,
    model_shard_count: usize,
}

impl Flux2KleinMlxComponents {
    pub(super) fn load_inner(&mut self) -> Result<Flux2KleinComponentLoad, String> {
        let mut attribution = self.new_attribution();
        let mut attribution_log = PerformanceAttributionLog::open(
            &self.performance_attribution_log_path,
            self.performance_attribution_enabled,
        )
        .map_err(|error| error.to_string())?;
        let Flux2KleinValidatedLoad {
            initialized_runtime,
            plan: load_plan,
        } = self.initialize_runtime_and_plan_flux2_klein_load(&mut attribution);
        let memory_snapshot = initialized_runtime
            .as_ref()
            .and_then(|initialized_runtime| initialized_runtime.memory_snapshot().ok());
        let failure_description = load_plan.as_ref().err().map(|failure| {
            failure
                .chars()
                .take(MAXIMUM_LOAD_FAILURE_DESCRIPTION_CHARACTERS)
                .collect::<String>()
        });
        let outcome = if load_plan.is_ok() {
            PerformanceAttributionOutcome::Success
        } else {
            PerformanceAttributionOutcome::Failed
        };
        if let Some(report) =
            attribution.finish_model_loading(ModelLoadingPerformanceAttributionMetadata {
                outcome,
                model_id: Some(self.serving_model_id.clone()),
                model_revision: Some(self.provenance.revision().to_owned()),
                prefill_transient_observation_completed: false,
                prefill_observed_transient_high_water_bytes: 0,
                total_artifact_payload_bytes: load_plan
                    .as_ref()
                    .ok()
                    .map(|load_plan| load_plan.total_artifact_payload_bytes),
                resident_model_payload_bytes: Some(0),
                model_shard_count: load_plan
                    .as_ref()
                    .ok()
                    .map(|load_plan| load_plan.model_shard_count),
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
        let Flux2KleinValidatedLoadPlan {
            validated_artifact,
            residency_plan,
            transformer_geometry,
            minimum_ceiling_bytes,
            memory_geometry,
            ..
        } = load_plan?;
        self.runtime = initialized_runtime;
        self.validated_artifact = Some(validated_artifact);
        self.residency_plan = Some(residency_plan);
        self.transformer_geometry = Some(transformer_geometry);
        self.memory_geometry = Some(memory_geometry);
        self.performance_attribution_log = Some(attribution_log);
        Ok(Flux2KleinComponentLoad::new(
            self.serving_model_id.clone(),
            self.provenance.revision(),
            official_capabilities(),
            minimum_ceiling_bytes,
        ))
    }

    /// Brings the MLX runtime up for this load — bounded initialization
    /// followed by an allocator-cache cleanup — and validates the artifact
    /// plus its memory admission only when that succeeded. Every step keeps
    /// its own measured region so load-time attribution keeps its original
    /// boundaries.
    fn initialize_runtime_and_plan_flux2_klein_load(
        &self,
        attribution: &mut PerformanceAttribution,
    ) -> Flux2KleinValidatedLoad {
        let initialized_runtime = match self.initialize_flux2_klein_runtime(attribution) {
            Ok(initialized_runtime) => Some(initialized_runtime),
            Err(load_failure) => {
                return Flux2KleinValidatedLoad {
                    initialized_runtime: None,
                    plan: Err(load_failure),
                };
            }
        };
        let load_plan = match self.validate_flux2_klein_load_plan(attribution) {
            Ok(load_plan) => load_plan,
            Err(load_failure) => {
                return Flux2KleinValidatedLoad {
                    initialized_runtime,
                    plan: Err(load_failure),
                };
            }
        };
        Flux2KleinValidatedLoad {
            initialized_runtime,
            plan: Ok(load_plan),
        }
    }

    /// Initializes the MLX runtime under this load's memory limits and clears
    /// the allocator cache; both steps sit inside their original
    /// `ImageRuntimeSetup` and `MlxAllocatorCacheCleanup` regions.
    fn initialize_flux2_klein_runtime(
        &self,
        attribution: &mut PerformanceAttribution,
    ) -> Result<MlxRuntime, String> {
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
        Ok(initialized_runtime)
    }

    /// Validates the artifact and plans its memory admission; the runtime is
    /// deliberately absent because none of these steps touch it.
    fn validate_flux2_klein_load_plan(
        &self,
        attribution: &mut PerformanceAttribution,
    ) -> Result<Flux2KleinValidatedLoadPlan, String> {
        let validated_artifact = Flux2KleinArtifactValidator::new()
            .validate_with_performance_attribution(
                &self.model_directory,
                self.provenance.clone(),
                attribution,
            )
            .map_err(|error| error.to_string())?;
        let geometry = memory_geometry(&validated_artifact)?;
        let total_artifact_payload_bytes = flux2_klein_total_artifact_payload_bytes(
            geometry.text_encoder_payload_bytes,
            validated_artifact.transformer_inventory().payload_bytes(),
            validated_artifact.vae_inventory().payload_bytes(),
        )?;
        let model_shard_count = flux2_klein_model_shard_count(
            validated_artifact.text_shard_count(),
            validated_artifact
                .transformer_inventory()
                .source_file_count(),
            validated_artifact.vae_inventory().source_file_count(),
        )?;
        let transformer_geometry = super::super::super::Flux2KleinTransformerGeometry::from_config(
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
        Ok(Flux2KleinValidatedLoadPlan {
            validated_artifact,
            residency_plan,
            transformer_geometry,
            minimum_ceiling_bytes,
            memory_geometry: geometry,
            total_artifact_payload_bytes,
            model_shard_count,
        })
    }
}

/// Sums the three component payload byte counts for the load report,
/// failing when `u64` addition would overflow the artifact accounting.
fn flux2_klein_total_artifact_payload_bytes(
    text_encoder_payload_bytes: u64,
    transformer_payload_bytes: u64,
    vae_payload_bytes: u64,
) -> Result<u64, String> {
    text_encoder_payload_bytes
        .checked_add(transformer_payload_bytes)
        .and_then(|payload_bytes| payload_bytes.checked_add(vae_payload_bytes))
        .ok_or_else(|| "FLUX.2 Klein artifact payload accounting overflowed".to_owned())
}

/// Sums the three component source-file counts for the load report, failing
/// when `usize` addition would overflow the artifact accounting.
fn flux2_klein_model_shard_count(
    text_shard_count: usize,
    transformer_source_file_count: usize,
    vae_source_file_count: usize,
) -> Result<usize, String> {
    text_shard_count
        .checked_add(transformer_source_file_count)
        .and_then(|shard_count| shard_count.checked_add(vae_source_file_count))
        .ok_or_else(|| "FLUX.2 Klein artifact shard accounting overflowed".to_owned())
}

#[cfg(test)]
#[path = "loading_tests.rs"]
mod loading_tests;
