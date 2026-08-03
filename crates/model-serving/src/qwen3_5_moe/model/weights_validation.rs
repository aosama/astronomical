use astronomical_runtime_integration::{MlxArray, MlxDtype};

use crate::{TensorDtype, TensorProfile};

use super::{Qwen3_5MoEConfig, Qwen3_5MoEExecutionError};

/// Validates one bound MLX tensor against its config-derived tensor profile.
pub(super) fn validate_bound_tensor(
    tensor_profile: &TensorProfile,
    bound_tensor: &MlxArray,
) -> Result<(), Qwen3_5MoEExecutionError> {
    let tensor_dtype_matches_profile = match tensor_profile.dtype {
        TensorDtype::BFloat16 => bound_tensor.dtype() == MlxDtype::BFloat16,
        TensorDtype::BFloat16OrFloat32 => {
            matches!(bound_tensor.dtype(), MlxDtype::BFloat16 | MlxDtype::Float32)
        }
        TensorDtype::Float32 => bound_tensor.dtype() == MlxDtype::Float32,
        TensorDtype::UInt32 => bound_tensor.dtype() == MlxDtype::UInt32,
    };
    if !tensor_dtype_matches_profile {
        return Err(Qwen3_5MoEExecutionError::InvalidTensor {
            tensor_name: tensor_profile.name.clone(),
            description: "tensor dtype differs from the config-derived profile",
        });
    }
    let expected_shape = tensor_profile
        .shape
        .iter()
        .map(|dimension| {
            i32::try_from(*dimension).map_err(|_| Qwen3_5MoEExecutionError::InvalidTensor {
                tensor_name: tensor_profile.name.clone(),
                description: "tensor dimension exceeds the MLX integer range",
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if bound_tensor.shape() != expected_shape {
        return Err(Qwen3_5MoEExecutionError::InvalidTensor {
            tensor_name: tensor_profile.name.clone(),
            description: "tensor shape differs from the config-derived profile",
        });
    }
    Ok(())
}

/// Validates that a quantized tensor profile uses an MLX-supported bit width.
pub(super) fn validate_quantized_tensor_bits(
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
    tensor_profile: &TensorProfile,
) -> Result<(), Qwen3_5MoEExecutionError> {
    let Some(module_name) = tensor_profile.name.strip_suffix(".scales") else {
        return Ok(());
    };
    let quantization_profile = qwen3_5_moe_config.quantization_profile_for_module(module_name);
    if quantization_profile.is_unquantized() {
        return Ok(());
    }
    if !matches!(quantization_profile.bits, 2 | 3 | 4 | 5 | 6 | 8) {
        return Err(Qwen3_5MoEExecutionError::InvalidTensor {
            tensor_name: tensor_profile.name.clone(),
            description: "quantized tensor bit width must be supported by MLX affine quantization",
        });
    }
    Ok(())
}
