use std::collections::HashSet;

use thiserror::Error;

/// Scalar type required by one persisted decoder-cache tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderCacheTensorDtype {
    BFloat16,
    Float32,
}

impl DecoderCacheTensorDtype {
    /// Returns the scalar payload width used by this decoder-cache tensor.
    #[must_use]
    pub const fn scalar_byte_count(self) -> usize {
        match self {
            Self::BFloat16 => 2,
            Self::Float32 => 4,
        }
    }
}

/// Default allocation granularity for append-only attention state.
pub const DEFAULT_APPEND_ONLY_ATTENTION_CAPACITY_GROWTH_TOKENS: usize = 256;

/// Static tensor contract for one named decoder-cache state component.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderCacheTensorLayout {
    qualified_role_name: String,
    dtype: DecoderCacheTensorDtype,
    dimensions: Vec<usize>,
    sequence_axis: Option<usize>,
}

/// One flattened tensor contract used by a decoder-state persistence file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderCachePersistedTensorLayout {
    decoder_layer_index: usize,
    tensor_layout: DecoderCacheTensorLayout,
}

impl DecoderCachePersistedTensorLayout {
    /// Returns the zero-based decoder-layer position.
    #[must_use]
    pub const fn decoder_layer_index(&self) -> usize {
        self.decoder_layer_index
    }

    /// Returns the tensor contract for this persisted decoder-state component.
    #[must_use]
    pub const fn tensor_layout(&self) -> &DecoderCacheTensorLayout {
        &self.tensor_layout
    }

    /// Returns the deterministic safetensors name for this component.
    #[must_use]
    pub fn persistent_tensor_name(&self) -> String {
        format!(
            "layer_{}_{}",
            self.decoder_layer_index,
            self.tensor_layout.qualified_role_name()
        )
    }
}

impl DecoderCacheTensorLayout {
    /// Creates a fixed-shape tensor restored only from a boundary snapshot.
    #[must_use]
    pub fn fixed(
        qualified_role_name: impl Into<String>,
        dtype: DecoderCacheTensorDtype,
        dimensions: Vec<usize>,
    ) -> Self {
        Self {
            qualified_role_name: qualified_role_name.into(),
            dtype,
            dimensions,
            sequence_axis: None,
        }
    }

    /// Creates a tensor sliced and concatenated along one token axis.
    #[must_use]
    pub fn sequence(
        qualified_role_name: impl Into<String>,
        dtype: DecoderCacheTensorDtype,
        dimensions: Vec<usize>,
        sequence_axis: usize,
    ) -> Self {
        Self {
            qualified_role_name: qualified_role_name.into(),
            dtype,
            dimensions,
            sequence_axis: Some(sequence_axis),
        }
    }

    /// Returns the deterministic model-qualified tensor role name.
    #[must_use]
    pub fn qualified_role_name(&self) -> &str {
        &self.qualified_role_name
    }

    /// Returns the exact scalar type expected in memory and on disk.
    #[must_use]
    pub const fn dtype(&self) -> DecoderCacheTensorDtype {
        self.dtype
    }

    /// Returns static tensor dimensions; a sequence dimension is represented by zero.
    #[must_use]
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    /// Returns the token axis for a sequence-sliceable tensor.
    #[must_use]
    pub const fn sequence_axis(&self) -> Option<usize> {
        self.sequence_axis
    }

    /// Returns the checked payload bytes for a fixed-shape tensor.
    pub fn fixed_payload_byte_count(&self) -> Result<usize, DecoderCacheLayoutError> {
        if self.sequence_axis.is_some() || self.dimensions.contains(&0) {
            return Err(DecoderCacheLayoutError::InvalidTensorPayloadGeometry {
                qualified_role_name: self.qualified_role_name.clone(),
                description: "a fixed tensor must not contain a sequence axis or dynamic dimension",
            });
        }
        checked_tensor_payload_byte_count(&self.qualified_role_name, &self.dimensions, self.dtype)
    }

    /// Returns checked payload bytes for one token of a sequence tensor.
    pub fn sequence_payload_byte_count_per_token(&self) -> Result<usize, DecoderCacheLayoutError> {
        let Some(sequence_axis) = self.sequence_axis else {
            return Err(DecoderCacheLayoutError::InvalidTensorPayloadGeometry {
                qualified_role_name: self.qualified_role_name.clone(),
                description: "a sequence tensor must declare a sequence axis",
            });
        };
        if sequence_axis >= self.dimensions.len() || self.dimensions[sequence_axis] != 0 {
            return Err(DecoderCacheLayoutError::InvalidTensorPayloadGeometry {
                qualified_role_name: self.qualified_role_name.clone(),
                description: "the sequence axis must contain the dynamic dimension",
            });
        }
        let mut one_token_dimensions = self.dimensions.clone();
        one_token_dimensions[sequence_axis] = 1;
        checked_tensor_payload_byte_count(
            &self.qualified_role_name,
            &one_token_dimensions,
            self.dtype,
        )
    }
}

fn checked_tensor_payload_byte_count(
    qualified_role_name: &str,
    dimensions: &[usize],
    dtype: DecoderCacheTensorDtype,
) -> Result<usize, DecoderCacheLayoutError> {
    let element_count = dimensions
        .iter()
        .try_fold(1_usize, |current_element_count, dimension| {
            current_element_count.checked_mul(*dimension)
        });
    let Some(element_count) = element_count else {
        return Err(DecoderCacheLayoutError::TensorPayloadByteCountOverflow {
            qualified_role_name: qualified_role_name.to_owned(),
        });
    };
    element_count
        .checked_mul(dtype.scalar_byte_count())
        .ok_or_else(|| DecoderCacheLayoutError::TensorPayloadByteCountOverflow {
            qualified_role_name: qualified_role_name.to_owned(),
        })
}

/// One exhaustive decoder-cache state family in a model layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecoderCacheLayerLayout {
    /// Append-only attention state, stored in token-sliceable blocks.
    AppendOnlyAttention {
        keys: DecoderCacheTensorLayout,
        values: DecoderCacheTensorLayout,
        capacity_growth_tokens: usize,
    },
    /// Fixed state restored only from the newest complete prompt boundary.
    RecurrentTensor { tensor: DecoderCacheTensorLayout },
    /// Ordered state components for hybrid decoder layers.
    Composite {
        components: Vec<DecoderCacheLayerLayout>,
    },
}

impl DecoderCacheLayerLayout {
    /// Defines append-only attention key/value state.
    #[must_use]
    pub fn append_only_attention(
        keys: DecoderCacheTensorLayout,
        values: DecoderCacheTensorLayout,
        capacity_growth_tokens: usize,
    ) -> Self {
        Self::AppendOnlyAttention {
            keys,
            values,
            capacity_growth_tokens,
        }
    }

    /// Defines one complete-boundary state tensor.
    #[must_use]
    pub fn recurrent_tensor(tensor: DecoderCacheTensorLayout) -> Self {
        Self::RecurrentTensor { tensor }
    }

    /// Defines an ordered hybrid layer state.
    #[must_use]
    pub fn composite(components: Vec<DecoderCacheLayerLayout>) -> Self {
        Self::Composite { components }
    }
}

/// Validated, architecture-neutral request decoder-cache layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderCacheLayout {
    layers: Vec<DecoderCacheLayerLayout>,
    sequence_tensor_count: usize,
    boundary_tensor_count: usize,
}

impl DecoderCacheLayout {
    /// Validates a model-owned decoder-cache layout before it creates live state or opens SSD files.
    pub fn new(layers: Vec<DecoderCacheLayerLayout>) -> Result<Self, DecoderCacheLayoutError> {
        let mut sequence_tensor_count = 0_usize;
        let mut boundary_tensor_count = 0_usize;
        for (layer_index, layer_layout) in layers.iter().enumerate() {
            let mut layer_tensor_role_names = HashSet::new();
            validate_layer_layout(
                layer_layout,
                layer_index,
                &mut layer_tensor_role_names,
                &mut sequence_tensor_count,
                &mut boundary_tensor_count,
            )?;
        }
        Ok(Self {
            layers,
            sequence_tensor_count,
            boundary_tensor_count,
        })
    }

    /// Returns the number of decoder layers with cache state.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns one model-owned layer layout by its decoder position.
    #[must_use]
    pub fn layer(&self, layer_index: usize) -> Option<&DecoderCacheLayerLayout> {
        self.layers.get(layer_index)
    }

    /// Returns the number of tensors persisted in every sequence-state block.
    #[must_use]
    pub const fn sequence_tensor_count(&self) -> usize {
        self.sequence_tensor_count
    }

    /// Returns the number of tensors persisted in one complete boundary snapshot.
    #[must_use]
    pub const fn boundary_tensor_count(&self) -> usize {
        self.boundary_tensor_count
    }

    /// Returns tensor contracts written to every sequence-state block.
    #[must_use]
    pub fn sequence_tensor_layouts(&self) -> Vec<DecoderCachePersistedTensorLayout> {
        self.persisted_tensor_layouts(true)
    }

    /// Returns tensor contracts written to one complete boundary-state snapshot.
    #[must_use]
    pub fn boundary_tensor_layouts(&self) -> Vec<DecoderCachePersistedTensorLayout> {
        self.persisted_tensor_layouts(false)
    }

    fn persisted_tensor_layouts(
        &self,
        include_sequence_tensors: bool,
    ) -> Vec<DecoderCachePersistedTensorLayout> {
        let expected_tensor_count = if include_sequence_tensors {
            self.sequence_tensor_count
        } else {
            self.boundary_tensor_count
        };
        let mut persisted_tensor_layouts = Vec::with_capacity(expected_tensor_count);
        for (decoder_layer_index, layer_layout) in self.layers.iter().enumerate() {
            collect_persisted_tensor_layouts(
                layer_layout,
                decoder_layer_index,
                include_sequence_tensors,
                &mut persisted_tensor_layouts,
            );
        }
        persisted_tensor_layouts
    }
}

fn collect_persisted_tensor_layouts(
    layer_layout: &DecoderCacheLayerLayout,
    decoder_layer_index: usize,
    include_sequence_tensors: bool,
    persisted_tensor_layouts: &mut Vec<DecoderCachePersistedTensorLayout>,
) {
    match layer_layout {
        DecoderCacheLayerLayout::AppendOnlyAttention { keys, values, .. } => {
            if include_sequence_tensors {
                persisted_tensor_layouts.push(DecoderCachePersistedTensorLayout {
                    decoder_layer_index,
                    tensor_layout: keys.clone(),
                });
                persisted_tensor_layouts.push(DecoderCachePersistedTensorLayout {
                    decoder_layer_index,
                    tensor_layout: values.clone(),
                });
            }
        }
        DecoderCacheLayerLayout::RecurrentTensor { tensor } => {
            if !include_sequence_tensors {
                persisted_tensor_layouts.push(DecoderCachePersistedTensorLayout {
                    decoder_layer_index,
                    tensor_layout: tensor.clone(),
                });
            }
        }
        DecoderCacheLayerLayout::Composite { components } => {
            for component_layout in components {
                collect_persisted_tensor_layouts(
                    component_layout,
                    decoder_layer_index,
                    include_sequence_tensors,
                    persisted_tensor_layouts,
                );
            }
        }
    }
}

fn validate_layer_layout(
    layer_layout: &DecoderCacheLayerLayout,
    layer_index: usize,
    layer_tensor_role_names: &mut HashSet<String>,
    sequence_tensor_count: &mut usize,
    boundary_tensor_count: &mut usize,
) -> Result<(), DecoderCacheLayoutError> {
    match layer_layout {
        DecoderCacheLayerLayout::AppendOnlyAttention {
            keys,
            values,
            capacity_growth_tokens,
        } => {
            if *capacity_growth_tokens == 0 {
                return Err(DecoderCacheLayoutError::ZeroCapacityGrowthTokens { layer_index });
            }
            validate_sequence_tensor(keys, layer_index, layer_tensor_role_names)?;
            validate_sequence_tensor(values, layer_index, layer_tensor_role_names)?;
            if keys.dtype != values.dtype
                || keys.dimensions != values.dimensions
                || keys.sequence_axis != values.sequence_axis
            {
                return Err(DecoderCacheLayoutError::AttentionTensorContractMismatch {
                    layer_index,
                });
            }
            *sequence_tensor_count = sequence_tensor_count.saturating_add(2);
        }
        DecoderCacheLayerLayout::RecurrentTensor { tensor } => {
            validate_boundary_tensor(tensor, layer_index, layer_tensor_role_names)?;
            *boundary_tensor_count = boundary_tensor_count.saturating_add(1);
        }
        DecoderCacheLayerLayout::Composite { components } => {
            if components.is_empty() {
                return Err(DecoderCacheLayoutError::EmptyComposite { layer_index });
            }
            for component_layout in components {
                validate_layer_layout(
                    component_layout,
                    layer_index,
                    layer_tensor_role_names,
                    sequence_tensor_count,
                    boundary_tensor_count,
                )?;
            }
        }
    }
    Ok(())
}

fn validate_sequence_tensor(
    tensor_layout: &DecoderCacheTensorLayout,
    layer_index: usize,
    layer_tensor_role_names: &mut HashSet<String>,
) -> Result<(), DecoderCacheLayoutError> {
    let Some(sequence_axis) = tensor_layout.sequence_axis else {
        return Err(DecoderCacheLayoutError::SequenceTensorMissingAxis {
            layer_index,
            qualified_role_name: tensor_layout.qualified_role_name.clone(),
        });
    };
    validate_tensor_role_and_dimensions(tensor_layout, layer_index, layer_tensor_role_names)?;
    if sequence_axis >= tensor_layout.dimensions.len() {
        return Err(DecoderCacheLayoutError::SequenceAxisOutsideTensorRank {
            layer_index,
            qualified_role_name: tensor_layout.qualified_role_name.clone(),
            sequence_axis,
            tensor_rank: tensor_layout.dimensions.len(),
        });
    }
    if tensor_layout.dimensions[sequence_axis] != 0 {
        return Err(
            DecoderCacheLayoutError::SequenceAxisMustUseDynamicDimension {
                layer_index,
                qualified_role_name: tensor_layout.qualified_role_name.clone(),
            },
        );
    }
    Ok(())
}

fn validate_boundary_tensor(
    tensor_layout: &DecoderCacheTensorLayout,
    layer_index: usize,
    layer_tensor_role_names: &mut HashSet<String>,
) -> Result<(), DecoderCacheLayoutError> {
    if tensor_layout.sequence_axis.is_some() {
        return Err(DecoderCacheLayoutError::BoundaryTensorHasSequenceAxis {
            layer_index,
            qualified_role_name: tensor_layout.qualified_role_name.clone(),
        });
    }
    validate_tensor_role_and_dimensions(tensor_layout, layer_index, layer_tensor_role_names)?;
    if tensor_layout.dimensions.contains(&0) {
        return Err(DecoderCacheLayoutError::BoundaryTensorHasDynamicDimension {
            layer_index,
            qualified_role_name: tensor_layout.qualified_role_name.clone(),
        });
    }
    Ok(())
}

fn validate_tensor_role_and_dimensions(
    tensor_layout: &DecoderCacheTensorLayout,
    layer_index: usize,
    layer_tensor_role_names: &mut HashSet<String>,
) -> Result<(), DecoderCacheLayoutError> {
    if tensor_layout.qualified_role_name.is_empty() {
        return Err(DecoderCacheLayoutError::EmptyTensorRole { layer_index });
    }
    if tensor_layout.dimensions.is_empty() {
        return Err(DecoderCacheLayoutError::ZeroTensorRank {
            layer_index,
            qualified_role_name: tensor_layout.qualified_role_name.clone(),
        });
    }
    if !layer_tensor_role_names.insert(tensor_layout.qualified_role_name.clone()) {
        return Err(DecoderCacheLayoutError::DuplicateTensorRole {
            layer_index,
            qualified_role_name: tensor_layout.qualified_role_name.clone(),
        });
    }
    Ok(())
}

/// One invalid model-owned decoder-cache layout.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum DecoderCacheLayoutError {
    #[error("model configuration {dimension_name} dimension does not fit usize")]
    ModelConfigurationDimensionOutsideUsizeRange { dimension_name: &'static str },
    #[error("decoder-cache layer {layer_index} has no composite components")]
    EmptyComposite { layer_index: usize },
    #[error("decoder-cache layer {layer_index} append-only attention has zero capacity growth")]
    ZeroCapacityGrowthTokens { layer_index: usize },
    #[error(
        "decoder-cache layer {layer_index} sequence tensor {qualified_role_name} has no sequence axis"
    )]
    SequenceTensorMissingAxis {
        layer_index: usize,
        qualified_role_name: String,
    },
    #[error(
        "decoder-cache layer {layer_index} tensor {qualified_role_name} has sequence axis {sequence_axis} outside rank {tensor_rank}"
    )]
    SequenceAxisOutsideTensorRank {
        layer_index: usize,
        qualified_role_name: String,
        sequence_axis: usize,
        tensor_rank: usize,
    },
    #[error(
        "decoder-cache layer {layer_index} sequence tensor {qualified_role_name} must use zero for its dynamic sequence dimension"
    )]
    SequenceAxisMustUseDynamicDimension {
        layer_index: usize,
        qualified_role_name: String,
    },
    #[error(
        "decoder-cache layer {layer_index} boundary tensor {qualified_role_name} must not have a sequence axis"
    )]
    BoundaryTensorHasSequenceAxis {
        layer_index: usize,
        qualified_role_name: String,
    },
    #[error(
        "decoder-cache layer {layer_index} boundary tensor {qualified_role_name} must not have a dynamic dimension"
    )]
    BoundaryTensorHasDynamicDimension {
        layer_index: usize,
        qualified_role_name: String,
    },
    #[error("decoder-cache layer {layer_index} has an empty tensor role")]
    EmptyTensorRole { layer_index: usize },
    #[error("decoder-cache layer {layer_index} tensor {qualified_role_name} has zero rank")]
    ZeroTensorRank {
        layer_index: usize,
        qualified_role_name: String,
    },
    #[error("decoder-cache layer {layer_index} repeats tensor role {qualified_role_name}")]
    DuplicateTensorRole {
        layer_index: usize,
        qualified_role_name: String,
    },
    #[error(
        "decoder-cache tensor {qualified_role_name} has invalid payload geometry: {description}"
    )]
    InvalidTensorPayloadGeometry {
        qualified_role_name: String,
        description: &'static str,
    },
    #[error("decoder-cache tensor {qualified_role_name} payload byte count overflowed")]
    TensorPayloadByteCountOverflow { qualified_role_name: String },
    #[error(
        "decoder-cache layer {layer_index} append-only attention keys and values have different contracts"
    )]
    AttentionTensorContractMismatch { layer_index: usize },
}
