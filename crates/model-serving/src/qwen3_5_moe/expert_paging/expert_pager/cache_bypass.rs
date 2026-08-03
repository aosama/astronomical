use crate::PerformanceAttribution;

use super::super::expert_cache::ExpertWeightMemoryCache;
use super::super::expert_cache_statistics::ExpertWeightMemoryCacheRequestReport;
use super::super::quantized_expert_manifest::QuantizedExpertPageManifest;
use super::{ExpertPager, ExpertPagingError, PagedExpertWeights};

impl ExpertPager {
    pub(super) fn load_selected_experts_while_bypassing_memory_cache(
        &self,
        runtime: &astronomical_runtime_integration::MlxRuntime,
        layer_index: usize,
        selected_expert_ids: &[usize],
        expert_weight_memory_cache: &mut ExpertWeightMemoryCache,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<
        (
            PagedExpertWeights,
            QuantizedExpertPageManifest,
            ExpertWeightMemoryCacheRequestReport,
        ),
        ExpertPagingError,
    > {
        let mut request_report = ExpertWeightMemoryCacheRequestReport::default();
        let (paged_expert_weights, direct_page_manifest, _) = self
            .load_selected_experts_with_performance_attribution(
                runtime,
                layer_index,
                selected_expert_ids,
                Some(expert_weight_memory_cache),
                performance_attribution,
            )?;
        request_report.cache_miss_count = selected_expert_ids.len();
        request_report.disk_page_load_count = selected_expert_ids.len();
        request_report.disk_batch_load_count = direct_page_manifest.source_manifests.len();
        expert_weight_memory_cache.record_cache_bypass_misses(request_report.cache_miss_count);
        expert_weight_memory_cache.record_disk_page_loads(request_report.disk_page_load_count);
        expert_weight_memory_cache.record_disk_batch_loads(request_report.disk_batch_load_count);
        Ok((paged_expert_weights, direct_page_manifest, request_report))
    }
}
