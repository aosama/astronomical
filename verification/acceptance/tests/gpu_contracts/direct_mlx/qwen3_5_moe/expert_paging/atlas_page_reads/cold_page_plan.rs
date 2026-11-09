//! Finds experts whose bytes are not in the file cache and plans disjoint
//! routed pages of them, so every measured read reaches the SSD.

use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use astronomical_model_serving::{
    QuantizedExpertLayerPlan, QuantizedExpertPageManifest,
    build_quantized_expert_page_manifest_from_plan,
};

use super::atlas_conversion::AtlasTestFile;
use super::progress_report::ProgressReporter;
use super::uncached_file_io::{any_vm_page_resident, open_uncached};

pub(super) struct PlannedPage {
    /// Position in `AtlasTestFile::layer_indexes`, not a model layer index.
    pub(super) atlas_layer_position: usize,
    pub(super) expert_ids: Vec<usize>,
    pub(super) stock_manifest: QuantizedExpertPageManifest,
}

/// Opens each shard and the atlas once for every residency check.
#[derive(Default)]
pub(super) struct ResidencyChecker {
    open_files_by_path: HashMap<PathBuf, File>,
}

impl ResidencyChecker {
    fn file(&mut self, path: &PathBuf) -> &File {
        self.open_files_by_path
            .entry(path.clone())
            .or_insert_with(|| open_uncached(path))
    }

    pub(super) fn stock_page_is_cached(
        &mut self,
        page_manifest: &QuantizedExpertPageManifest,
    ) -> bool {
        for shard_manifest in &page_manifest.source_manifests {
            let shard_file = self.file(&shard_manifest.source_file);
            for interval in &shard_manifest.source_intervals {
                if any_vm_page_resident(
                    shard_file,
                    interval.source_file_offset,
                    interval.source_byte_count as u64,
                ) {
                    return true;
                }
            }
        }
        false
    }

    pub(super) fn atlas_page_is_cached(
        &mut self,
        atlas: &AtlasTestFile,
        atlas_layer_position: usize,
        expert_ids: &[usize],
    ) -> bool {
        let atlas_file = self.file(&atlas.path);
        expert_ids.iter().any(|&expert_id| {
            any_vm_page_resident(
                atlas_file,
                atlas.record_offset(atlas_layer_position, expert_id),
                atlas.record_stride_bytes,
            )
        })
    }
}

/// For every model layer, the experts whose stock bytes are entirely uncached.
pub(super) fn cold_stock_expert_ids_by_layer(
    layer_plans: &[QuantizedExpertLayerPlan],
    expert_count: usize,
    residency_checker: &mut ResidencyChecker,
) -> Vec<Vec<usize>> {
    let mut progress = ProgressReporter::new("cold-expert-scan", layer_plans.len());
    let scan_started_at = Instant::now();
    let mut cold_expert_ids_by_layer = Vec::with_capacity(layer_plans.len());
    for (layer_index, layer_plan) in layer_plans.iter().enumerate() {
        let mut cold_expert_ids = Vec::new();
        for expert_id in 0..expert_count {
            let single_expert_manifest =
                build_quantized_expert_page_manifest_from_plan(layer_plan, &[expert_id])
                    .expect("a one-expert manifest builds");
            if !residency_checker.stock_page_is_cached(&single_expert_manifest) {
                cold_expert_ids.push(expert_id);
            }
        }
        cold_expert_ids_by_layer.push(cold_expert_ids);
        progress.record(layer_index + 1, scan_started_at);
    }
    cold_expert_ids_by_layer
}

/// The `layer_count` model layers with the most cold experts, ascending.
pub(super) fn coldest_layer_indexes(
    cold_expert_ids_by_layer: &[Vec<usize>],
    layer_count: usize,
) -> Vec<usize> {
    let mut layer_indexes_by_coldness: Vec<usize> = (0..cold_expert_ids_by_layer.len()).collect();
    layer_indexes_by_coldness.sort_by_key(|&layer_index| {
        (
            std::cmp::Reverse(cold_expert_ids_by_layer[layer_index].len()),
            layer_index,
        )
    });
    let mut chosen_layer_indexes: Vec<usize> = layer_indexes_by_coldness
        .into_iter()
        .take(layer_count)
        .collect();
    chosen_layer_indexes.sort_unstable();
    chosen_layer_indexes
}

/// Disjoint pages of cold experts, interleaved across the atlas layers.
pub(super) fn plan_cold_pages(
    layer_plans: &[QuantizedExpertLayerPlan],
    cold_expert_ids_by_layer: &[Vec<usize>],
    atlas: &AtlasTestFile,
    experts_per_page: usize,
) -> Vec<PlannedPage> {
    let mut xorshift_state = 0x5eed_0002_u64;
    let pages_by_atlas_layer: Vec<Vec<Vec<usize>>> = atlas
        .layer_indexes
        .iter()
        .map(|&layer_index| {
            let mut cold_expert_ids = cold_expert_ids_by_layer[layer_index].clone();
            shuffle_expert_ids(&mut cold_expert_ids, &mut xorshift_state);
            cold_expert_ids
                .chunks_exact(experts_per_page)
                .map(ascending_expert_ids)
                .collect()
        })
        .collect();
    let deepest_page_count = pages_by_atlas_layer.iter().map(Vec::len).max().unwrap_or(0);
    let mut planned_pages = Vec::new();
    for page_index in 0..deepest_page_count {
        for (atlas_layer_position, layer_pages) in pages_by_atlas_layer.iter().enumerate() {
            let Some(expert_ids) = layer_pages.get(page_index) else {
                continue;
            };
            let layer_plan = &layer_plans[atlas.layer_indexes[atlas_layer_position]];
            planned_pages.push(PlannedPage {
                stock_manifest: build_quantized_expert_page_manifest_from_plan(
                    layer_plan, expert_ids,
                )
                .expect("the page manifest builds"),
                atlas_layer_position,
                expert_ids: expert_ids.clone(),
            });
        }
    }
    planned_pages
}

/// The production page manifest requires ascending expert ids.
fn ascending_expert_ids(expert_ids: &[usize]) -> Vec<usize> {
    let mut ascending = expert_ids.to_vec();
    ascending.sort_unstable();
    ascending
}

fn shuffle_expert_ids(expert_ids: &mut [usize], xorshift_state: &mut u64) {
    for swap_position in (1..expert_ids.len()).rev() {
        *xorshift_state ^= *xorshift_state << 13;
        *xorshift_state ^= *xorshift_state >> 7;
        *xorshift_state ^= *xorshift_state << 17;
        expert_ids.swap(
            swap_position,
            (*xorshift_state % (swap_position as u64 + 1)) as usize,
        );
    }
}
