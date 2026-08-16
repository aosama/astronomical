use std::collections::{BTreeMap, btree_map::Entry};

use super::{
    raw_tensor_name_parser::{
        LagunaExpertSourcePackaging, LagunaRawExpertProjection, LagunaRawTensorNameParser,
        LagunaRawTensorNamespace, ParsedLagunaTensorName,
    },
    tensor_assembly::{LagunaRawTensorNameRecord, LagunaTensorAssembly},
    tensor_id::{
        LagunaExpertProjection, LagunaLayerTensorRole, LagunaTensorComponent, LagunaTensorId,
    },
    tensor_name_contract::{LagunaExpertGateUpLayout, LagunaTensorNameContract},
    tensor_name_error::LagunaTensorNameNormalizationError,
};

const MAX_TENSOR_NAME_COUNT: usize = 1_000_000;
const MAX_TENSOR_NAME_BYTES: usize = 1_024;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ExpertTensorKey {
    layer_index: usize,
    projection: LagunaRawExpertProjection,
    component: LagunaTensorComponent,
}

enum ExpertSourceGroup {
    Stacked(String),
    PerExpert(BTreeMap<usize, String>),
}

impl ExpertSourceGroup {
    const fn packaging(&self) -> LagunaExpertSourcePackaging {
        match self {
            Self::Stacked(_) => LagunaExpertSourcePackaging::Stacked,
            Self::PerExpert(_) => LagunaExpertSourcePackaging::PerExpert,
        }
    }

    fn first_source_name(&self) -> &str {
        match self {
            Self::Stacked(raw_name) => raw_name,
            Self::PerExpert(raw_names_by_expert) => raw_names_by_expert
                .first_key_value()
                .map(|(_, raw_name)| raw_name.as_str())
                .unwrap_or("<missing expert source>"),
        }
    }
}

/// Builds a strict canonical Laguna tensor contract from neutral raw-name records.
pub struct LagunaTensorNameNormalizer {
    layer_count: usize,
    expert_count: usize,
}

impl LagunaTensorNameNormalizer {
    /// Configures synthetic or normalized geometry without model-name assumptions.
    #[must_use]
    pub const fn new(layer_count: usize, expert_count: usize) -> Self {
        Self {
            layer_count,
            expert_count,
        }
    }

    /// Canonicalizes aliases and packaging before downstream binding can inspect names.
    pub fn normalize(
        &self,
        raw_tensor_records: &[LagunaRawTensorNameRecord],
    ) -> Result<LagunaTensorNameContract, LagunaTensorNameNormalizationError> {
        if self.layer_count == 0 {
            return Err(LagunaTensorNameNormalizationError::InvalidLayerCount);
        }
        if raw_tensor_records.len() > MAX_TENSOR_NAME_COUNT {
            return Err(
                LagunaTensorNameNormalizationError::TensorInventoryTooLarge {
                    actual_count: raw_tensor_records.len(),
                    maximum_count: MAX_TENSOR_NAME_COUNT,
                },
            );
        }

        let parser = LagunaRawTensorNameParser::new(self.layer_count, self.expert_count);
        let mut selected_namespace = None;
        let mut assemblies = BTreeMap::new();
        let mut expert_source_groups = BTreeMap::new();
        for raw_tensor_record in raw_tensor_records {
            let raw_name = raw_tensor_record.raw_name();
            validate_raw_name_bounds(raw_name)?;
            let parsed_name = parser.parse(raw_name)?;
            validate_namespace(&mut selected_namespace, parsed_name.namespace())?;
            match parsed_name {
                ParsedLagunaTensorName::Direct { tensor_id, .. } => {
                    insert_assembly(
                        &mut assemblies,
                        tensor_id,
                        LagunaTensorAssembly::direct(raw_name),
                    )?;
                }
                ParsedLagunaTensorName::RoutedExpert {
                    layer_index,
                    projection,
                    component,
                    packaging,
                    expert_index,
                    ..
                } => {
                    if self.expert_count == 0 {
                        return Err(
                            LagunaTensorNameNormalizationError::ExpertTensorWithoutExperts {
                                layer_index,
                            },
                        );
                    }
                    insert_expert_source(
                        &mut expert_source_groups,
                        ExpertTensorKey {
                            layer_index,
                            projection,
                            component,
                        },
                        packaging,
                        expert_index,
                        raw_name,
                    )?;
                }
            }
        }

        let expert_gate_up_layouts =
            validate_expert_sources(&expert_source_groups, self.expert_count)?;
        build_expert_assemblies(&mut assemblies, expert_source_groups)?;
        Ok(LagunaTensorNameContract::new(
            assemblies,
            expert_gate_up_layouts,
        ))
    }
}

fn validate_raw_name_bounds(raw_name: &str) -> Result<(), LagunaTensorNameNormalizationError> {
    if raw_name.is_empty() {
        return Err(LagunaTensorNameNormalizationError::EmptyTensorName);
    }
    if raw_name.len() > MAX_TENSOR_NAME_BYTES {
        return Err(LagunaTensorNameNormalizationError::TensorNameTooLong {
            actual_bytes: raw_name.len(),
            maximum_bytes: MAX_TENSOR_NAME_BYTES,
        });
    }
    Ok(())
}

fn validate_namespace(
    selected_namespace: &mut Option<LagunaRawTensorNamespace>,
    current_namespace: LagunaRawTensorNamespace,
) -> Result<(), LagunaTensorNameNormalizationError> {
    match selected_namespace {
        Some(expected_namespace) if *expected_namespace != current_namespace => {
            Err(LagunaTensorNameNormalizationError::MixedTensorNamespaces)
        }
        Some(_) => Ok(()),
        None => {
            *selected_namespace = Some(current_namespace);
            Ok(())
        }
    }
}

fn insert_expert_source(
    expert_source_groups: &mut BTreeMap<ExpertTensorKey, ExpertSourceGroup>,
    tensor_key: ExpertTensorKey,
    packaging: LagunaExpertSourcePackaging,
    expert_index: Option<usize>,
    raw_name: &str,
) -> Result<(), LagunaTensorNameNormalizationError> {
    match expert_source_groups.entry(tensor_key) {
        Entry::Vacant(vacant_entry) => {
            let source_group = match (packaging, expert_index) {
                (LagunaExpertSourcePackaging::Stacked, None) => {
                    ExpertSourceGroup::Stacked(raw_name.to_owned())
                }
                (LagunaExpertSourcePackaging::PerExpert, Some(expert_index)) => {
                    ExpertSourceGroup::PerExpert(BTreeMap::from([(
                        expert_index,
                        raw_name.to_owned(),
                    )]))
                }
                _ => return Err(unknown_internal_packaging(raw_name)),
            };
            vacant_entry.insert(source_group);
            Ok(())
        }
        Entry::Occupied(mut occupied_entry) => match occupied_entry.get_mut() {
            ExpertSourceGroup::PerExpert(raw_names_by_expert)
                if packaging == LagunaExpertSourcePackaging::PerExpert =>
            {
                let expert_index =
                    expert_index.ok_or_else(|| unknown_internal_packaging(raw_name))?;
                if let Some(first_source_name) = raw_names_by_expert.get(&expert_index) {
                    return Err(collision_error(tensor_key, first_source_name, raw_name));
                }
                raw_names_by_expert.insert(expert_index, raw_name.to_owned());
                Ok(())
            }
            existing_group if existing_group.packaging() != packaging => {
                Err(LagunaTensorNameNormalizationError::MixedExpertPackaging {
                    layer_index: tensor_key.layer_index,
                })
            }
            existing_group => Err(collision_error(
                tensor_key,
                existing_group.first_source_name(),
                raw_name,
            )),
        },
    }
}

fn validate_expert_sources(
    expert_source_groups: &BTreeMap<ExpertTensorKey, ExpertSourceGroup>,
    expert_count: usize,
) -> Result<BTreeMap<usize, LagunaExpertGateUpLayout>, LagunaTensorNameNormalizationError> {
    let mut projections_by_layer_component =
        BTreeMap::<(usize, LagunaTensorComponent), Vec<LagunaRawExpertProjection>>::new();
    let mut packaging_by_layer = BTreeMap::new();
    for (tensor_key, source_group) in expert_source_groups {
        if let ExpertSourceGroup::PerExpert(raw_names_by_expert) = source_group
            && raw_names_by_expert.len() != expert_count
        {
            return Err(LagunaTensorNameNormalizationError::IncompleteExpertSet {
                layer_index: tensor_key.layer_index,
                projection: tensor_key.projection.canonical_projection(),
                component: tensor_key.component,
                expected_expert_count: expert_count,
                actual_expert_count: raw_names_by_expert.len(),
            });
        }
        match packaging_by_layer.entry(tensor_key.layer_index) {
            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(source_group.packaging());
            }
            Entry::Occupied(occupied_entry)
                if *occupied_entry.get() != source_group.packaging() =>
            {
                return Err(LagunaTensorNameNormalizationError::MixedExpertPackaging {
                    layer_index: tensor_key.layer_index,
                });
            }
            Entry::Occupied(_) => {}
        }
        projections_by_layer_component
            .entry((tensor_key.layer_index, tensor_key.component))
            .or_default()
            .push(tensor_key.projection);
    }

    let mut layouts_by_layer = BTreeMap::new();
    for ((layer_index, component), projections) in projections_by_layer_component {
        let has_gate = projections.contains(&LagunaRawExpertProjection::Gate);
        let has_up = projections.contains(&LagunaRawExpertProjection::Up);
        let has_gate_up = projections.contains(&LagunaRawExpertProjection::GateUp);
        let has_down = projections.contains(&LagunaRawExpertProjection::Down);
        let layout = match (has_gate, has_up, has_gate_up, has_down) {
            (true, true, false, true) => LagunaExpertGateUpLayout::Split,
            (false, false, true, true) => LagunaExpertGateUpLayout::Fused,
            (true, _, true, _) | (_, true, true, _) => {
                return Err(
                    LagunaTensorNameNormalizationError::MixedExpertGateUpLayouts { layer_index },
                );
            }
            _ => {
                return Err(
                    LagunaTensorNameNormalizationError::IncompleteExpertProjectionSet {
                        layer_index,
                        component,
                    },
                );
            }
        };
        match layouts_by_layer.entry(layer_index) {
            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(layout);
            }
            Entry::Occupied(occupied_entry) if *occupied_entry.get() != layout => {
                return Err(
                    LagunaTensorNameNormalizationError::MixedExpertGateUpLayouts { layer_index },
                );
            }
            Entry::Occupied(_) => {}
        }
    }
    Ok(layouts_by_layer)
}

fn build_expert_assemblies(
    assemblies: &mut BTreeMap<LagunaTensorId, LagunaTensorAssembly>,
    expert_source_groups: BTreeMap<ExpertTensorKey, ExpertSourceGroup>,
) -> Result<(), LagunaTensorNameNormalizationError> {
    for (tensor_key, source_group) in expert_source_groups {
        match tensor_key.projection {
            LagunaRawExpertProjection::GateUp => {
                for projection in [LagunaExpertProjection::Gate, LagunaExpertProjection::Up] {
                    let assembly = build_source_assembly(&source_group, Some(projection));
                    insert_assembly(
                        assemblies,
                        routed_expert_id(tensor_key, projection),
                        assembly,
                    )?;
                }
            }
            raw_projection => {
                let projection = raw_projection.canonical_projection();
                let assembly = build_source_assembly(&source_group, None);
                insert_assembly(
                    assemblies,
                    routed_expert_id(tensor_key, projection),
                    assembly,
                )?;
            }
        }
    }
    Ok(())
}

fn build_source_assembly(
    source_group: &ExpertSourceGroup,
    fused_projection: Option<LagunaExpertProjection>,
) -> LagunaTensorAssembly {
    match (source_group, fused_projection) {
        (ExpertSourceGroup::Stacked(raw_name), None) => LagunaTensorAssembly::stacked(raw_name),
        (ExpertSourceGroup::PerExpert(raw_names_by_expert), None) => {
            LagunaTensorAssembly::per_expert(raw_names_by_expert.values().cloned().collect())
        }
        (ExpertSourceGroup::Stacked(raw_name), Some(projection)) => {
            LagunaTensorAssembly::fused_stacked(raw_name, projection)
        }
        (ExpertSourceGroup::PerExpert(raw_names_by_expert), Some(projection)) => {
            LagunaTensorAssembly::fused_per_expert(
                raw_names_by_expert.values().cloned().collect(),
                projection,
            )
        }
    }
}

fn insert_assembly(
    assemblies: &mut BTreeMap<LagunaTensorId, LagunaTensorAssembly>,
    tensor_id: LagunaTensorId,
    assembly: LagunaTensorAssembly,
) -> Result<(), LagunaTensorNameNormalizationError> {
    if let Some(existing_assembly) = assemblies.get(&tensor_id) {
        let first_source_name = existing_assembly
            .sources()
            .first()
            .map(|source| source.raw_name())
            .unwrap_or("<missing source>");
        let conflicting_source_name = assembly
            .sources()
            .first()
            .map(|source| source.raw_name())
            .unwrap_or("<missing source>");
        return Err(LagunaTensorNameNormalizationError::CanonicalCollision {
            tensor_id,
            first_source_name: first_source_name.to_owned(),
            conflicting_source_name: conflicting_source_name.to_owned(),
        });
    }
    assemblies.insert(tensor_id, assembly);
    Ok(())
}

fn collision_error(
    tensor_key: ExpertTensorKey,
    first_source_name: &str,
    conflicting_source_name: &str,
) -> LagunaTensorNameNormalizationError {
    LagunaTensorNameNormalizationError::CanonicalCollision {
        tensor_id: routed_expert_id(tensor_key, tensor_key.projection.canonical_projection()),
        first_source_name: first_source_name.to_owned(),
        conflicting_source_name: conflicting_source_name.to_owned(),
    }
}

const fn routed_expert_id(
    tensor_key: ExpertTensorKey,
    projection: LagunaExpertProjection,
) -> LagunaTensorId {
    LagunaTensorId::Layer {
        layer_index: tensor_key.layer_index,
        role: LagunaLayerTensorRole::RoutedExpert(projection),
        component: tensor_key.component,
    }
}

fn unknown_internal_packaging(raw_name: &str) -> LagunaTensorNameNormalizationError {
    // This branch protects construction invariants if the private parser changes later.
    LagunaTensorNameNormalizationError::UnknownTensorName {
        tensor_name: raw_name.to_owned(),
    }
}
