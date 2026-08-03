use crate::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, mlx_runtime::check_status, raw};

impl MlxRuntime {
    /// Samples one categorical index along `axis` from logits using a deterministic MLX key seed.
    pub fn categorical_sample(
        &self,
        logits: &MlxArray,
        axis: i32,
        seed: u64,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_categorical_arguments(logits, axis)?;
        let random_key = self.random_key(seed)?;
        self.categorical_sample_with_key(logits, axis, &random_key)
    }

    /// Samples one categorical index using an explicit split PRNG key.
    pub fn categorical_sample_with_key(
        &self,
        logits: &MlxArray,
        axis: i32,
        random_key: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_categorical_arguments(logits, axis)?;
        validate_random_key(random_key)?;
        self.output_array("sample MLX categorical logits", |output, stream| {
            // SAFETY: Logits, key, and stream are live; axis was validated; output
            // is uniquely writable for MLX to populate.
            unsafe {
                raw::mlx_random_categorical(output, logits.raw(), axis, random_key.raw(), stream)
            }
        })
    }

    /// Creates the same two-word PRNG state as `mlx.core.random.seed`.
    pub fn random_key(&self, seed: u64) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("create MLX random key", |output, _stream| {
            // SAFETY: The output handle is uniquely writable and the seed is copied by value.
            unsafe { raw::mlx_random_key(output, seed) }
        })
    }

    /// Advances one PRNG state and returns its next state plus one sample key.
    pub fn split_random_key(
        &self,
        random_state: &MlxArray,
    ) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
        validate_random_key(random_state)?;
        let mut next_random_state = MlxArray::empty();
        let mut sample_key = MlxArray::empty();
        // SAFETY: Input and stream are live, and both distinct output handles
        // are uniquely writable for the duration of the split call.
        let status = unsafe {
            raw::mlx_random_split(
                next_random_state.raw_mut(),
                sample_key.raw_mut(),
                random_state.raw(),
                self.gpu_stream().raw(),
            )
        };
        check_status(status, "split an MLX random key")?;
        next_random_state.require_populated("split an MLX random key")?;
        sample_key.require_populated("split an MLX random key")?;
        Ok((next_random_state, sample_key))
    }
}

fn validate_categorical_arguments(logits: &MlxArray, axis: i32) -> Result<(), MlxRuntimeError> {
    const OPERATION: &str = "sample MLX categorical logits";
    if !matches!(
        logits.dtype(),
        MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::BFloat16
    ) {
        return Err(runtime_operation_error(
            OPERATION,
            "categorical logits must have a supported floating dtype",
        ));
    }
    let rank = i32::try_from(logits.shape().len()).map_err(|_| {
        runtime_operation_error(OPERATION, "categorical logits rank exceeds i32 range")
    })?;
    if rank == 0 || axis < -rank || axis >= rank {
        return Err(runtime_operation_error(
            OPERATION,
            "categorical axis must refer to an existing logits dimension",
        ));
    }
    Ok(())
}

fn validate_random_key(random_key: &MlxArray) -> Result<(), MlxRuntimeError> {
    if random_key.dtype() != MlxDtype::UInt32 || random_key.shape() != [2] {
        return Err(runtime_operation_error(
            "use an MLX random key",
            "random keys must be uint32 arrays with shape [2]",
        ));
    }
    Ok(())
}

fn runtime_operation_error(operation: &'static str, description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: description.to_owned(),
    }
}
