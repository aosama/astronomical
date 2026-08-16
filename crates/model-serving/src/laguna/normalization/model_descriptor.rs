/// Supported execution dtype declared by Laguna configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagunaExecutionDtype {
    Float16,
    Bfloat16,
    Float32,
}

/// Validated model-wide Laguna geometry and canonical scalar behavior.
#[derive(Clone, Debug, PartialEq)]
pub struct LagunaModelDescriptor {
    vocabulary_size: u32,
    hidden_size: u32,
    dense_intermediate_size: u32,
    layer_count: usize,
    maximum_position_count: u32,
    rms_norm_epsilon: f64,
    execution_dtype: LagunaExecutionDtype,
    has_tied_embeddings: bool,
    router_logit_softcap: f64,
}

impl LagunaModelDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        vocabulary_size: u32,
        hidden_size: u32,
        dense_intermediate_size: u32,
        layer_count: usize,
        maximum_position_count: u32,
        rms_norm_epsilon: f64,
        execution_dtype: LagunaExecutionDtype,
        has_tied_embeddings: bool,
        router_logit_softcap: f64,
    ) -> Self {
        Self {
            vocabulary_size,
            hidden_size,
            dense_intermediate_size,
            layer_count,
            maximum_position_count,
            rms_norm_epsilon,
            execution_dtype,
            has_tied_embeddings,
            router_logit_softcap,
        }
    }

    /// Returns the tokenizer vocabulary size.
    #[must_use]
    pub const fn vocabulary_size(&self) -> u32 {
        self.vocabulary_size
    }

    /// Returns the decoder hidden width.
    #[must_use]
    pub const fn hidden_size(&self) -> u32 {
        self.hidden_size
    }

    /// Returns the dense SwiGLU intermediate width.
    #[must_use]
    pub const fn dense_intermediate_size(&self) -> u32 {
        self.dense_intermediate_size
    }

    /// Returns the validated decoder layer count.
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    /// Returns the maximum combined prompt and generated position count.
    #[must_use]
    pub const fn maximum_position_count(&self) -> u32 {
        self.maximum_position_count
    }

    /// Returns the positive finite root-mean-square normalization epsilon.
    #[must_use]
    pub const fn rms_norm_epsilon(&self) -> f64 {
        self.rms_norm_epsilon
    }

    /// Returns the arithmetic dtype used by model execution.
    #[must_use]
    pub const fn execution_dtype(&self) -> LagunaExecutionDtype {
        self.execution_dtype
    }

    /// Returns whether input and output embeddings are tied.
    #[must_use]
    pub const fn has_tied_embeddings(&self) -> bool {
        self.has_tied_embeddings
    }

    /// Returns zero when router softcapping is disabled, otherwise its positive finite cap.
    #[must_use]
    pub const fn router_logit_softcap(&self) -> f64 {
        self.router_logit_softcap
    }
}
