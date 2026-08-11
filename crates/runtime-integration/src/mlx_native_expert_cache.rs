//! Safe Rust ownership around Astronomical's native demand-only expert cache.
//!
//! Rust supplies startup-validated file ranges and model memory policy. The
//! native owner performs route synchronization, direct range reads, layer-balanced
//! least-recently-used eviction, immutable page-table publication, and custom
//! Metal projection construction. Every returned snapshot has independent
//! lifetime ownership so lazy MLX graphs cannot observe an evicted page.

use std::{ffi::CString, os::unix::ffi::OsStrExt, path::PathBuf, ptr::NonNull};

use crate::{
    MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError,
    mlx_native_expert_cache_report::{
        MlxNativeExpertCacheRequestReport, MlxNativeExpertCacheStatistics, request_report_from_raw,
        statistics_from_raw, zero_raw_request_report,
    },
    mlx_runtime::{
        check_status, classify_mlx_error, clear_captured_mlx_error, take_captured_mlx_error,
    },
    raw,
};

const NATIVE_CACHE_OPERATION: &str = "manage the native MLX expert cache";
const NATIVE_GATHER_OPERATION: &str = "build a native paged expert gathered product";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
/// Stable projection indices shared with the C and Metal page-table layouts.
pub enum MlxNativeExpertProjection {
    Gate = 0,
    Up = 1,
    Down = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
/// Stable affine parameter indices shared with the native descriptor contract.
pub enum MlxNativeExpertParameter {
    PackedWeight = 0,
    Scales = 1,
    Biases = 2,
}

#[derive(Clone, Debug)]
/// One complete safetensors tensor split into equal per-expert source ranges.
pub struct MlxNativeExpertTensorSourceDescriptor {
    projection: MlxNativeExpertProjection,
    parameter: MlxNativeExpertParameter,
    quantization_group_size: i32,
    quantization_bits: i32,
    source_file_path: PathBuf,
    tensor_payload_offset: u64,
    bytes_per_expert: usize,
    expert_shape: Vec<i32>,
    dtype: MlxDtype,
}

impl MlxNativeExpertTensorSourceDescriptor {
    #[must_use]
    pub fn new(
        projection: MlxNativeExpertProjection,
        parameter: MlxNativeExpertParameter,
        quantization_group_size: i32,
        quantization_bits: i32,
        source_file_path: PathBuf,
        tensor_payload_offset: u64,
        bytes_per_expert: usize,
        expert_shape: Vec<i32>,
        dtype: MlxDtype,
    ) -> Self {
        Self {
            projection,
            parameter,
            quantization_group_size,
            quantization_bits,
            source_file_path,
            tensor_payload_offset,
            bytes_per_expert,
            expert_shape,
            dtype,
        }
    }
}

#[derive(Clone, Debug)]
/// Startup inventory for every tensor source belonging to one sparse layer.
pub struct MlxNativeExpertLayerDescriptor {
    expert_capacity: usize,
    tensor_sources: Vec<MlxNativeExpertTensorSourceDescriptor>,
}

impl MlxNativeExpertLayerDescriptor {
    #[must_use]
    pub fn new(
        expert_capacity: usize,
        tensor_sources: Vec<MlxNativeExpertTensorSourceDescriptor>,
    ) -> Self {
        Self {
            expert_capacity,
            tensor_sources,
        }
    }
}

#[derive(Debug)]
/// Unique owner of one mutable native cache.
///
/// Methods take `&self` because mutation is encapsulated by the opaque C++
/// owner. The model-serving engine invokes them serially for one active request;
/// this type does not expose a concurrent cache-policy API.
pub struct MlxNativeExpertCache {
    native_cache: NonNull<raw::astronomical_native_expert_cache>,
}

impl MlxNativeExpertCache {
    /// Copies validated layer metadata into a native cache after MLX startup.
    ///
    /// `runtime` is intentionally part of the construction contract even though
    /// the native constructor needs no raw stream: callers must initialize MLX's
    /// error handler, device, and Metal environment before kernel ownership is
    /// created.
    pub fn new(
        _runtime: &MlxRuntime,
        layer_descriptors: &[MlxNativeExpertLayerDescriptor],
        maximum_resident_payload_byte_count: u64,
    ) -> Result<Self, MlxRuntimeError> {
        // C strings, shape vectors, and nested descriptor vectors live through
        // the constructor call. C++ copies all metadata and opens source files
        // before returning, so the resulting cache borrows no Rust allocation.
        let source_file_paths = layer_descriptors
            .iter()
            .flat_map(|layer_descriptor| &layer_descriptor.tensor_sources)
            .map(|tensor_source| {
                CString::new(tensor_source.source_file_path.as_os_str().as_bytes()).map_err(|_| {
                    MlxRuntimeError::RuntimeOperation {
                        operation: NATIVE_CACHE_OPERATION,
                        description: "expert source path contains an interior NUL byte".to_owned(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut source_file_path_index = 0usize;
        let raw_tensor_sources_by_layer = layer_descriptors
            .iter()
            .map(|layer_descriptor| {
                layer_descriptor
                    .tensor_sources
                    .iter()
                    .map(|tensor_source| {
                        let raw_source = raw::astronomical_native_expert_tensor_source {
                            projection_index: tensor_source.projection as i32,
                            parameter_index: tensor_source.parameter as i32,
                            quantization_group_size: tensor_source.quantization_group_size,
                            quantization_bits: tensor_source.quantization_bits,
                            source_file_path: source_file_paths[source_file_path_index].as_ptr(),
                            tensor_payload_offset: tensor_source.tensor_payload_offset,
                            bytes_per_expert: tensor_source.bytes_per_expert,
                            expert_shape: tensor_source.expert_shape.as_ptr(),
                            expert_shape_dimension_count: tensor_source.expert_shape.len(),
                            dtype: tensor_source.dtype.to_raw(),
                        };
                        source_file_path_index += 1;
                        raw_source
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let raw_layer_descriptors = layer_descriptors
            .iter()
            .zip(&raw_tensor_sources_by_layer)
            .map(|(layer_descriptor, raw_tensor_sources)| {
                raw::astronomical_native_expert_layer_descriptor {
                    expert_capacity: layer_descriptor.expert_capacity,
                    tensor_sources: raw_tensor_sources.as_ptr(),
                    tensor_source_count: raw_tensor_sources.len(),
                }
            })
            .collect::<Vec<_>>();
        clear_captured_mlx_error();
        // SAFETY: All nested descriptor storage and C strings remain live for the
        // copying native constructor. The returned pointer enters unique ownership.
        let native_cache = unsafe {
            raw::astronomical_native_expert_cache_new(
                raw_layer_descriptors.as_ptr(),
                raw_layer_descriptors.len(),
                maximum_resident_payload_byte_count,
            )
        };
        NonNull::new(native_cache)
            .map(|native_cache| Self { native_cache })
            .ok_or_else(|| captured_native_error(NATIVE_CACHE_OPERATION))
    }

    /// Synchronizes an ordinary route, updates policy, loads misses, and returns
    /// an immutable generation snapshot plus request-local diagnostics.
    pub fn prepare_layer(
        &self,
        runtime: &MlxRuntime,
        layer_index: usize,
        selected_expert_indices: &MlxArray,
        collect_performance_metrics: bool,
    ) -> Result<
        (
            MlxNativeExpertCacheSnapshot,
            MlxNativeExpertCacheRequestReport,
        ),
        MlxRuntimeError,
    > {
        let mut native_snapshot = std::ptr::null_mut();
        let mut raw_report = zero_raw_request_report();
        clear_captured_mlx_error();
        // SAFETY: The engine serializes cache operations, input handles remain
        // live, and both output pointers are uniquely writable for this call.
        let prepare_status = unsafe {
            raw::astronomical_native_expert_cache_prepare_layer(
                self.native_cache.as_ptr(),
                layer_index,
                selected_expert_indices.raw(),
                runtime.gpu_stream().raw(),
                i32::from(collect_performance_metrics),
                &mut native_snapshot,
                &mut raw_report,
            )
        };
        check_status(prepare_status, NATIVE_CACHE_OPERATION)?;
        let native_snapshot = NonNull::new(native_snapshot)
            .ok_or_else(|| captured_native_error(NATIVE_CACHE_OPERATION))?;
        Ok((
            MlxNativeExpertCacheSnapshot { native_snapshot },
            request_report_from_raw(raw_report),
        ))
    }

    /// Phase 1: turns repeated router assignments into an exact page list.
    ///
    /// This may wait for lazy router work, but it does not read expert weights,
    /// allocate an expert page, or evict a retained page. The caller can safely
    /// inspect the returned byte counts and perform memory admission first.
    pub fn analyze_layer(
        &self,
        runtime: &MlxRuntime,
        layer_index: usize,
        selected_expert_indices: &MlxArray,
        collect_performance_metrics: bool,
    ) -> Result<
        (
            MlxNativeExpertCacheRouteAnalysis,
            MlxNativeExpertCacheRequestReport,
        ),
        MlxRuntimeError,
    > {
        let mut native_route_analysis = std::ptr::null_mut();
        let mut raw_report = zero_raw_request_report();
        clear_captured_mlx_error();
        // SAFETY: The engine serializes cache operations, input handles remain
        // live, and both output pointers are uniquely writable for this call.
        let analysis_status = unsafe {
            raw::astronomical_native_expert_cache_analyze_layer(
                self.native_cache.as_ptr(),
                layer_index,
                selected_expert_indices.raw(),
                runtime.gpu_stream().raw(),
                i32::from(collect_performance_metrics),
                &mut native_route_analysis,
                &mut raw_report,
            )
        };
        check_status(analysis_status, NATIVE_CACHE_OPERATION)?;
        let native_route_analysis = NonNull::new(native_route_analysis)
            .ok_or_else(|| captured_native_error(NATIVE_CACHE_OPERATION))?;
        Ok((
            MlxNativeExpertCacheRouteAnalysis {
                native_route_analysis,
            },
            request_report_from_raw(raw_report),
        ))
    }

    /// Phase 2 using the current ceiling: loads exact misses and publishes a snapshot.
    pub fn commit_layer(
        &self,
        runtime: &MlxRuntime,
        route_analysis: MlxNativeExpertCacheRouteAnalysis,
        collect_performance_metrics: bool,
    ) -> Result<
        (
            MlxNativeExpertCacheSnapshot,
            MlxNativeExpertCacheRequestReport,
        ),
        MlxRuntimeError,
    > {
        self.commit_layer_with_maximum_resident_payload_byte_count(
            runtime,
            route_analysis,
            self.statistics().maximum_resident_payload_byte_count(),
            collect_performance_metrics,
        )
    }

    /// Phase 2 using a new ceiling chosen from the post-analysis memory sample.
    ///
    /// "Atomically" means native code changes the ceiling and commits the route
    /// as one uninterrupted cache operation. No other policy step can remove a
    /// selected cache hit between those actions.
    pub fn commit_layer_with_maximum_resident_payload_byte_count(
        &self,
        runtime: &MlxRuntime,
        route_analysis: MlxNativeExpertCacheRouteAnalysis,
        maximum_resident_payload_byte_count: u64,
        collect_performance_metrics: bool,
    ) -> Result<
        (
            MlxNativeExpertCacheSnapshot,
            MlxNativeExpertCacheRequestReport,
        ),
        MlxRuntimeError,
    > {
        let mut native_snapshot = std::ptr::null_mut();
        let mut raw_report = zero_raw_request_report();
        clear_captured_mlx_error();
        // SAFETY: The engine serializes cache operations, the route analysis and
        // stream remain live through the synchronous commit, and output pointers
        // are uniquely writable.
        let commit_status = unsafe {
            raw::astronomical_native_expert_cache_commit_layer(
                self.native_cache.as_ptr(),
                route_analysis.native_route_analysis.as_ptr(),
                maximum_resident_payload_byte_count,
                runtime.gpu_stream().raw(),
                i32::from(collect_performance_metrics),
                &mut native_snapshot,
                &mut raw_report,
            )
        };
        check_status(commit_status, NATIVE_CACHE_OPERATION)?;
        let native_snapshot = NonNull::new(native_snapshot)
            .ok_or_else(|| captured_native_error(NATIVE_CACHE_OPERATION))?;
        Ok((
            MlxNativeExpertCacheSnapshot { native_snapshot },
            request_report_from_raw(raw_report),
        ))
    }

    pub fn update_maximum_resident_payload_byte_count(
        &self,
        maximum_resident_payload_byte_count: u64,
    ) -> Result<(), MlxRuntimeError> {
        clear_captured_mlx_error();
        // SAFETY: The cache is uniquely borrowed for this synchronous policy update.
        let update_status = unsafe {
            raw::astronomical_native_expert_cache_update_maximum_resident_payload_bytes(
                self.native_cache.as_ptr(),
                maximum_resident_payload_byte_count,
            )
        };
        check_status(update_status, NATIVE_CACHE_OPERATION)
    }

    #[must_use]
    pub fn freeze_retention_growth(&self) -> bool {
        // SAFETY: This owner always holds a valid native cache pointer.
        unsafe {
            raw::astronomical_native_expert_cache_freeze_retention_growth(
                self.native_cache.as_ptr(),
            ) != 0
        }
    }

    pub fn reclaim_retained_payload_bytes(
        &self,
        reclamation_target_byte_count: u64,
    ) -> Result<bool, MlxRuntimeError> {
        self.run_boolean_policy_operation(|native_cache, output_changed| unsafe {
            raw::astronomical_native_expert_cache_reclaim_retained_payload_bytes(
                native_cache,
                reclamation_target_byte_count,
                output_changed,
            )
        })
    }

    #[must_use]
    pub fn resume_retention_growth(&self) -> bool {
        // SAFETY: This owner always holds a valid native cache pointer.
        unsafe {
            raw::astronomical_native_expert_cache_resume_retention_growth(
                self.native_cache.as_ptr(),
            ) != 0
        }
    }

    fn run_boolean_policy_operation(
        &self,
        operation: impl FnOnce(*mut raw::astronomical_native_expert_cache, *mut i32) -> i32,
    ) -> Result<bool, MlxRuntimeError> {
        let mut did_change = 0;
        clear_captured_mlx_error();
        let operation_status = operation(self.native_cache.as_ptr(), &mut did_change);
        check_status(operation_status, NATIVE_CACHE_OPERATION)?;
        Ok(did_change != 0)
    }

    #[must_use]
    pub fn statistics(&self) -> MlxNativeExpertCacheStatistics {
        // SAFETY: The cache owner remains live for this copying scalar query.
        statistics_from_raw(unsafe {
            raw::astronomical_native_expert_cache_get_statistics(self.native_cache.as_ptr())
        })
    }
}

impl Drop for MlxNativeExpertCache {
    fn drop(&mut self) {
        // SAFETY: This owner releases its native cache exactly once.
        unsafe { raw::astronomical_native_expert_cache_free(self.native_cache.as_ptr()) };
    }
}

#[derive(Debug)]
/// Exact native route evidence held between memory admission and cache commit.
/// It owns no newly loaded expert payload and releases its native handle on drop.
pub struct MlxNativeExpertCacheRouteAnalysis {
    native_route_analysis: NonNull<raw::astronomical_native_expert_route_analysis>,
}

impl Drop for MlxNativeExpertCacheRouteAnalysis {
    fn drop(&mut self) {
        // SAFETY: This owner releases its native route analysis exactly once.
        unsafe {
            raw::astronomical_native_expert_route_analysis_free(self.native_route_analysis.as_ptr())
        };
    }
}

#[derive(Debug)]
/// Immutable native page-table generation retained by one or more lazy products.
///
/// C++ stores shared page owners beside the Metal address table.
pub struct MlxNativeExpertCacheSnapshot {
    native_snapshot: NonNull<raw::astronomical_native_expert_snapshot>,
}

impl MlxNativeExpertCacheSnapshot {
    /// Builds one gate, up, or down gathered product without stacking experts.
    ///
    /// Graph construction does not evaluate the product. The native primitive
    /// captures this snapshot's shared owner, so dropping the Rust wrapper after
    /// this call cannot invalidate pages needed by later evaluation.
    #[allow(clippy::too_many_arguments)]
    pub fn gather_matmul(
        &self,
        runtime: &MlxRuntime,
        projection: MlxNativeExpertProjection,
        activations: &MlxArray,
        selected_expert_indices: &MlxArray,
        transpose_weights: bool,
        sorted_indices: bool,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let mut output = MlxArray::empty();
        clear_captured_mlx_error();
        // SAFETY: The immutable snapshot and input handles remain live through
        // graph construction, and the output handle is uniquely writable. The
        // native primitive takes its own shared snapshot owner for lazy use.
        let operation_status = unsafe {
            raw::astronomical_native_expert_snapshot_gather_matmul(
                output.raw_mut(),
                self.native_snapshot.as_ptr(),
                projection as i32,
                activations.raw(),
                selected_expert_indices.raw(),
                i32::from(transpose_weights),
                i32::from(sorted_indices),
                runtime.gpu_stream().raw(),
            )
        };
        check_status(operation_status, NATIVE_GATHER_OPERATION)?;
        output.require_populated(NATIVE_GATHER_OPERATION)?;
        Ok(output)
    }
}

impl Drop for MlxNativeExpertCacheSnapshot {
    fn drop(&mut self) {
        // SAFETY: This owner releases its immutable native snapshot exactly once.
        unsafe { raw::astronomical_native_expert_snapshot_free(self.native_snapshot.as_ptr()) };
    }
}

fn captured_native_error(operation: &'static str) -> MlxRuntimeError {
    let description = take_captured_mlx_error()
        .unwrap_or_else(|| "native expert cache returned no owner".to_owned());
    classify_mlx_error(operation, description)
}
