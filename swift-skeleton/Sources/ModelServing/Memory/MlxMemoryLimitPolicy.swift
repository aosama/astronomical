import Foundation;

import MLX;

/// Applies the worker's effective ceiling to both MLX controls: active
/// graph-evaluation memory and reclaimable allocator-cache retention.
public enum MlxMemoryLimitPolicy {

    /// Applies one validated positive effective memory ceiling in bytes.
    public static func apply(effectiveCeilingBytes: UInt64) -> Void {
        let effectiveMemoryLimitBytes: Int = Int(clamping: effectiveCeilingBytes);
        Memory.memoryLimit = effectiveMemoryLimitBytes;
        Memory.cacheLimit = effectiveMemoryLimitBytes;
    }

    /// Waits for submitted default-GPU work and returns reclaimable buffers
    /// before a replacement model competes for the same MLX process budget.
    public static func prepareForModelLoad(effectiveCeilingBytes: UInt64) -> Void {
        Stream.gpu.synchronize();
        Memory.clearCache();
        MlxMemoryLimitPolicy.apply(effectiveCeilingBytes: effectiveCeilingBytes);
    }
}
