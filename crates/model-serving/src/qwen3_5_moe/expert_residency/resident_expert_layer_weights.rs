use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights;

/// Complete contiguous sparse-expert arrays for one Qwen3.5 layer.
///
/// Every projection keeps the artifact's leading expert axis. Consequently the
/// router's global expert identifiers index these arrays directly, with no page
/// table or compact-slot remapping between routing and matrix multiplication.
#[derive(Debug)]
pub(crate) struct Qwen3_5ResidentExpertLayerWeights {
    pub(crate) gate_projection: Qwen3_5AffineWeights,
    pub(crate) up_projection: Qwen3_5AffineWeights,
    pub(crate) down_projection: Qwen3_5AffineWeights,
}

impl Qwen3_5ResidentExpertLayerWeights {
    pub(super) fn new(
        gate_projection: Qwen3_5AffineWeights,
        up_projection: Qwen3_5AffineWeights,
        down_projection: Qwen3_5AffineWeights,
    ) -> Self {
        Self {
            gate_projection,
            up_projection,
            down_projection,
        }
    }

    pub(crate) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        // Materialization evaluates packed weights and affine companions together
        // so publication cannot expose a projection with lazy source reads left.
        self.gate_projection.append_array_references(arrays);
        self.up_projection.append_array_references(arrays);
        self.down_projection.append_array_references(arrays);
    }
}
