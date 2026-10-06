import Foundation;

import Metal;

/// Machine GPU facts the worker's memory policy derives from.
///
/// Replaces the MLX-C device-info query of
/// crates/runtime-integration/src/mlx_runtime_device_info.rs: the
/// recommended working set the MLX runtime reports is the same fact the
/// Metal device exposes natively, so the Swift runtime asks Metal directly
/// instead of linking a C binding for it.
public enum MlxRuntime {

    public enum MlxRuntimeError: Error, Equatable {
        /// No default GPU device is available to this process.
        case gpuDeviceUnavailable;
    }

    /// The maximum recommended GPU working-set size in bytes for this
    /// machine's default GPU, as the GPU stack itself reports it. The value
    /// adapts to the machine; nothing about it is hardwired.
    public static func maximumRecommendedGpuWorkingSetSizeBytes() throws -> Int {
        guard let gpuDevice: (any MTLDevice) = MTLCreateSystemDefaultDevice() else {
            throw MlxRuntimeError.gpuDeviceUnavailable;
        }
        return Int(gpuDevice.recommendedMaxWorkingSetSize);
    }
}
