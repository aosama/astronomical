import Foundation;

import Testing;

import MLX;

import ModelServing;

@Suite
final class MlxMemoryLimitPolicyTests {

    @Test(.timeLimit(.minutes(1)))
    func should_apply_the_same_effective_limit_to_active_memory_and_allocator_cache() throws {
        let originalActiveMemoryLimitBytes: Int = Memory.memoryLimit;
        let originalAllocatorCacheLimitBytes: Int = Memory.cacheLimit;
        defer {
            Memory.memoryLimit = originalActiveMemoryLimitBytes;
            Memory.cacheLimit = originalAllocatorCacheLimitBytes;
            Memory.clearCache();
        }
        let requestedEffectiveCeilingBytes: UInt64 = 64_000_000;

        MlxMemoryLimitPolicy.apply(effectiveCeilingBytes: requestedEffectiveCeilingBytes);

        #expect(Memory.memoryLimit == Int(requestedEffectiveCeilingBytes));
        #expect(Memory.cacheLimit == Int(requestedEffectiveCeilingBytes));
    }
}
