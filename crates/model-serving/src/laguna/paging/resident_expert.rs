//! One Laguna expert as the decode-cache weight unit, plus split/stack helpers.

use astronomical_runtime_integration::MlxRuntime;

use crate::expert_paging::QuantizedExpertPageManifest;
use crate::laguna::model::LagunaBoundLinear;
use crate::memory::ResidentExpertWeight;

use super::error::LagunaPagingError;
use super::weight_page::{LagunaExpertWeightPage, LagunaGateUpPage};

/// One expert's fused-or-split gate/up plus down projection.
#[derive(Debug)]
pub struct LagunaResidentExpert {
    gate_up: LagunaGateUpPage,
    down: LagunaBoundLinear,
    payload_bytes: u64,
}

impl ResidentExpertWeight for LagunaResidentExpert {
    fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
}

impl LagunaGateUpPage {
    fn slice_leading_axis(
        &self,
        runtime: &MlxRuntime,
        start: i32,
        stop: i32,
    ) -> Result<Self, LagunaPagingError> {
        match self {
            Self::Split { gate, up } => Ok(Self::Split {
                gate: gate.slice_leading_axis(runtime, start, stop)?,
                up: up.slice_leading_axis(runtime, start, stop)?,
            }),
            Self::Fused(fused_gate_up) => Ok(Self::Fused(
                fused_gate_up.slice_leading_axis(runtime, start, stop)?,
            )),
        }
    }

    fn concatenate_leading_axis(
        runtime: &MlxRuntime,
        pages: &[&Self],
    ) -> Result<Self, LagunaPagingError> {
        let Some(first) = pages.first() else {
            return Err(LagunaPagingError::PageExecution {
                description: "decode stacking requires at least one gate/up page",
            });
        };
        match first {
            Self::Fused(_) => {
                let fused = pages
                    .iter()
                    .map(|page| match page {
                        Self::Fused(fused_gate_up) => Ok(fused_gate_up),
                        Self::Split { .. } => Err(LagunaPagingError::PageExecution {
                            description: "decode stacking cannot mix fused and split gate/up",
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Fused(LagunaBoundLinear::concatenate_leading_axis(
                    runtime, &fused,
                )?))
            }
            Self::Split { .. } => {
                let mut gates = Vec::new();
                let mut ups = Vec::new();
                for page in pages {
                    match page {
                        Self::Split { gate, up } => {
                            gates.push(gate);
                            ups.push(up);
                        }
                        Self::Fused(_) => {
                            return Err(LagunaPagingError::PageExecution {
                                description: "decode stacking cannot mix fused and split gate/up",
                            });
                        }
                    }
                }
                Ok(Self::Split {
                    gate: LagunaBoundLinear::concatenate_leading_axis(runtime, &gates)?,
                    up: LagunaBoundLinear::concatenate_leading_axis(runtime, &ups)?,
                })
            }
        }
    }
}

impl LagunaExpertWeightPage {
    /// Slices each selected expert into an independently owned decode-cache unit.
    pub(in crate::laguna) fn split_resident_experts(
        &self,
        runtime: &MlxRuntime,
        bytes_per_expert: u64,
    ) -> Result<Vec<(usize, LagunaResidentExpert)>, LagunaPagingError> {
        let expert_count = i32::try_from(self.manifest.expert_ids.len()).map_err(|_| {
            LagunaPagingError::PageExecution {
                description: "resident expert count exceeds MLX slice range",
            }
        })?;
        let mut resident_experts = Vec::with_capacity(self.manifest.expert_ids.len());
        for (slot, expert_id) in self.manifest.expert_ids.iter().copied().enumerate() {
            let start = i32::try_from(slot).map_err(|_| LagunaPagingError::PageExecution {
                description: "resident expert slot exceeds MLX slice range",
            })?;
            if start >= expert_count {
                break;
            }
            resident_experts.push((
                expert_id,
                LagunaResidentExpert {
                    gate_up: self.gate_up.slice_leading_axis(runtime, start, start + 1)?,
                    down: self.down.slice_leading_axis(runtime, start, start + 1)?,
                    payload_bytes: bytes_per_expert,
                },
            ));
        }
        Ok(resident_experts)
    }

    /// Stacks decode-resident experts into one gatherable page for SwiGLU.
    pub(in crate::laguna) fn stack_resident_experts(
        runtime: &MlxRuntime,
        expert_capacity: usize,
        resident_weights: &[(usize, &LagunaResidentExpert)],
    ) -> Result<Self, LagunaPagingError> {
        if resident_weights.is_empty() {
            return Err(LagunaPagingError::PageExecution {
                description: "decode stacking requires at least one resident expert",
            });
        }
        let mut ordered = resident_weights.to_vec();
        ordered.sort_by_key(|(expert_id, _)| *expert_id);
        let expert_ids = ordered
            .iter()
            .map(|(expert_id, _)| *expert_id)
            .collect::<Vec<_>>();
        let gate_up_pages = ordered
            .iter()
            .map(|(_, expert)| &expert.gate_up)
            .collect::<Vec<_>>();
        let down_projections = ordered
            .iter()
            .map(|(_, expert)| &expert.down)
            .collect::<Vec<_>>();
        let payload_byte_count = ordered
            .iter()
            .map(|(_, expert)| expert.payload_bytes)
            .fold(0_u64, u64::saturating_add);
        Ok(Self {
            manifest: stacked_decode_manifest(expert_ids, expert_capacity, payload_byte_count)?,
            gate_up: LagunaGateUpPage::concatenate_leading_axis(runtime, &gate_up_pages)?,
            down: LagunaBoundLinear::concatenate_leading_axis(runtime, &down_projections)?,
        })
    }
}

fn stacked_decode_manifest(
    expert_ids: Vec<usize>,
    expert_capacity: usize,
    payload_byte_count: u64,
) -> Result<QuantizedExpertPageManifest, LagunaPagingError> {
    let mut page_slot_by_global_expert_id = vec![u32::MAX; expert_capacity];
    for (page_slot, expert_id) in expert_ids.iter().copied().enumerate() {
        let slot = u32::try_from(page_slot).map_err(|_| LagunaPagingError::PageExecution {
            description: "decode stack slot exceeds UInt32",
        })?;
        if expert_id >= expert_capacity {
            return Err(LagunaPagingError::PageExecution {
                description: "decode stack expert id exceeds layer capacity",
            });
        }
        page_slot_by_global_expert_id[expert_id] = slot;
    }
    Ok(QuantizedExpertPageManifest {
        expert_ids,
        page_slot_by_global_expert_id,
        source_manifests: Vec::new(),
        payload_byte_count,
    })
}
