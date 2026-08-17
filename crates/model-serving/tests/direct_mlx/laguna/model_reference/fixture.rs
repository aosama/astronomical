//! Constructs production Laguna models and independently retained reference tensors.

use std::collections::{BTreeSet, HashMap};

use astronomical_model_serving::{
    LagunaModel, LagunaNativeWeights, LagunaTargetNormalizer, LagunaTensorId,
};
use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use super::rows::ReferenceRow;
use super::tensor_fixture::build_tensor_inventories;

pub(super) struct ReferenceFixture {
    pub(super) model: LagunaModel,
    pub(super) reference_tensors: HashMap<LagunaTensorId, MlxArray>,
    pub(super) observed_affine_profiles: BTreeSet<(i32, i32)>,
}

pub(super) fn build_fixture(runtime: &MlxRuntime, row: &ReferenceRow) -> ReferenceFixture {
    let contract = LagunaTargetNormalizer::normalize(
        &serde_json::to_vec(&row.target_config).expect("reference config should serialize"),
    )
    .unwrap_or_else(|error| panic!("{} should normalize: {error}", row.row_name));
    let inventories = build_tensor_inventories(runtime, &contract, row);
    let weights = LagunaNativeWeights::bind(runtime, inventories.production_tensors, &contract)
        .unwrap_or_else(|error| panic!("{} weights should bind: {error:?}", row.row_name));
    let model = LagunaModel::new(contract, weights)
        .unwrap_or_else(|error| panic!("{} model should construct: {error:?}", row.row_name));
    ReferenceFixture {
        model,
        reference_tensors: inventories.reference_tensors,
        observed_affine_profiles: inventories.observed_affine_profiles,
    }
}
