//! Bindgen configuration for the narrow MLX and Astronomical native surface.
//!
//! Explicit allowlists keep generated Rust bindings small and prevent private
//! upstream APIs from becoming accidental production dependencies. Native
//! comments are disabled because ownership documentation belongs on the safe
//! Rust wrappers and hand-authored C header rather than generated output.

use std::{error::Error, path::Path};

const BINDGEN_FUNCTION_ALLOWLIST: &str = concat!(
    "mlx_(set_error_handler|metal_set_metallib_path|version|string_(new|data|free)|clear_cache|",
    "device_(new_type|free)|device_info_(new|get|get_size|free)|",
    "compile|closure_(new|new_func|free|apply)|",
    "get_(active_memory|cache_memory|memory_limit|peak_memory)|reset_peak_memory|set_(cache|memory)_limit|",
    "array_(new|new_data|free|eval|shape|ndim|dtype|size|nbytes|data_(float32|uint32)|item_uint32|set)|",
    "default_(cpu|gpu)_stream_new|stream_free|synchronize|",
    "(add(mm)?|arange|argmax_axis|argpartition_axis|argsort_axis|astype|broadcast_to|concatenate_axis|contiguous|conv(1d|3d)|cos|cumsum|dequantize|divide|erf|exp|expand_dims|floor_divide|gather_(mm|qmm)|greater|greater_equal|log1p|logaddexp|matmul|power|",
    "max_axis|multiply|negative|put_along_axis|quantized_matmul|repeat_axis|reshape|sigmoid|sin|slice(_update)?|softmax_axis|subtract|sum_axis|tanh|",
    "squeeze_axis|stack_axis|take_along_axis|take_axis|topk_axis|transpose_axes|where|zeros)|",
    "fast_(rms_norm|layer_norm|rope(_dynamic)?|scaled_dot_product_attention)|fast_metal_kernel(_config)?_(new|free|apply|add_output_arg|set_grid|set_thread_group|set_init_value|add_template_arg_(dtype|int|bool))|random_(categorical|key|split)|eval|async_eval|",
    "vector_array_(new|new_data|free|get|size|set_value)|vector_string_(new_data|free)|io_(reader|writer)_(new|free)|",
    "save_safetensors_writer|",
    "load_safetensors_reader|map_string_to_array_(new|free|get)|",
    "paged_(buffer_slot_(new|commit|is_committed|view|free)|file_reader_(new|free))|read_paged_buffer_ranges|",
    "map_string_to_array_insert|map_string_to_string_(new|free|insert))|",
    "astronomical_metal_expert_loader_(start|wait|free)|",
    "astronomical_native_expert_(cache_(new|prepare_layer|update_maximum_resident_payload_bytes|freeze_retention_growth|reclaim_retained_payload_bytes|resume_retention_growth|get_statistics|free)|snapshot_(gather_matmul|free))",
);

const BINDGEN_TYPE_ALLOWLIST: &str = concat!(
    "(mlx_(error_handler_func|string|array|dtype|stream|closure|device|device_info|device_type|io_reader|io_vtable|",
    "optional_(dtype|float|int)|vector_array|vector_string|fast_metal_kernel(_config)?|map_string_to_array|map_string_to_string|paged_buffer_slot|paged_file_reader|paged_buffer_read_range)|",
    "astronomical_metal_expert_loader_(output_tensor|load_range|metrics|handle)|",
    "astronomical_native_expert_(cache|snapshot|projection|parameter|tensor_source|layer_descriptor|cache_request_report|cache_statistics))",
);

pub fn generate_bindings(
    mlx_c_source_directory: &Path,
    manifest_directory: &Path,
    output_directory: &Path,
    should_build_experimental_aligned_expert_packs: bool,
) -> Result<(), Box<dyn Error>> {
    let mlx_c_header = mlx_c_source_directory.join("mlx/c/mlx.h");
    let astronomical_native_expert_cache_header = manifest_directory
        .join("native")
        .join("expert_cache/astronomical_native_expert_cache.h");
    let astronomical_metal_expert_loader_header = manifest_directory
        .join("native")
        .join("experimental/aligned_expert_packs/astronomical_metal_expert_loader.h");
    let astronomical_metal_expert_loader_directory = astronomical_metal_expert_loader_header
        .parent()
        .ok_or("Astronomical Metal expert loader header has no parent directory")?;
    require_file(&mlx_c_header, "MLX C umbrella header")?;
    require_file(
        &astronomical_native_expert_cache_header,
        "Astronomical native expert cache header",
    )?;
    let mut bindings_builder = bindgen::Builder::default()
        .header(mlx_c_header.to_string_lossy())
        .header(astronomical_native_expert_cache_header.to_string_lossy())
        .clang_arg(format!(
            "-I{}",
            astronomical_native_expert_cache_header
                .parent()
                .ok_or("native expert cache header has no parent")?
                .display()
        ))
        .clang_arg(format!("-I{}", mlx_c_source_directory.display()))
        .allowlist_function(BINDGEN_FUNCTION_ALLOWLIST)
        .allowlist_type(BINDGEN_TYPE_ALLOWLIST);
    if should_build_experimental_aligned_expert_packs {
        require_file(
            &astronomical_metal_expert_loader_header,
            "experimental Astronomical Metal expert loader header",
        )?;
        bindings_builder = bindings_builder
            .header(astronomical_metal_expert_loader_header.to_string_lossy())
            .clang_arg(format!(
                "-I{}",
                astronomical_metal_expert_loader_directory.display()
            ));
    }
    let bindings = bindings_builder.generate_comments(false).generate()?;
    bindings.write_to_file(output_directory.join("mlx_c_bindings.rs"))?;
    Ok(())
}

fn require_file(file_path: &Path, description: &str) -> Result<(), Box<dyn Error>> {
    if !file_path.is_file() {
        return Err(format!("missing {description} at {}", file_path.display()).into());
    }
    Ok(())
}
